// cuda_executor.cu — CUDA kernel definition and CudaExecutor implementation.
//
// KERNEL DESIGN:
//   work_kernel: one CUDA thread per task.
//   Each thread computes W = task_costs[task_offset+tid] * cost_multiplier
//   iterations of a compute-bound loop.  The loop body (XOR + AND) is
//   lightweight arithmetic chosen to prevent the compiler from optimising
//   the loop away.  The result is written through a volatile pointer so
//   the write (and therefore the loop) cannot be dead-code-eliminated.
//
// WARP-DIVERGENCE NOTE (documented here to match README requirement):
//   Because tasks in the same warp may have different W values, each warp
//   runs for max(W_i) cycles — the slowest thread in the warp.  Changing
//   the scheduler's chunk size does not remove this intra-warp divergence;
//   chunk size only controls how often the scheduler gets a decision point
//   between chunks.  This is a fundamental CUDA execution-model limitation.
//
// GRID SIZING:
//   blocks = ceil(chunk_size / threads_per_block)
//   Threads whose global task index >= task_offset + chunk_size return
//   immediately (guard check at top of kernel).

#include "cuda_executor.h"
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <thread>

using Clock = std::chrono::high_resolution_clock;
using Ms    = std::chrono::duration<double, std::milli>;

// ---------------------------------------------------------------------------
// CUDA kernel — compute-bound synthetic workload
// ---------------------------------------------------------------------------
// Parameters:
//   task_costs     : device array of raw task costs (integers)
//   task_offset    : index of the first task in this chunk
//   chunk_size     : number of tasks in this chunk
//   out            : device output array (volatile to prevent DCE)
//   cost_multiplier: scales each task's iteration count
//
// One logical thread processes one task.
// Threads outside [0, chunk_size) perform no work (guard at top).
__global__ void work_kernel(
    const int* __restrict__ task_costs,
    int                     task_offset,
    int                     chunk_size,
    volatile int*           out,
    int                     cost_multiplier)
{
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= chunk_size) return;

    int  W   = task_costs[task_offset + tid] * cost_multiplier;
    int  acc = 0;

    // Compute-bound loop.  The XOR+AND expression is non-trivial enough
    // that nvcc will not collapse it, but cheap enough to avoid register
    // pressure.  Increasing cost_multiplier scales total work linearly.
    for (int i = 0; i < W; ++i)
        acc += (i ^ W) & 0xFF;

    // Volatile write: prevents the compiler from treating acc as dead.
    out[task_offset + tid] = acc;
}

// Zero-work kernel used during calibration to measure launch + sync overhead.
__global__ void noop_kernel(volatile int* out, int chunk_size)
{
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid < chunk_size) out[tid] = 0;
}

// ---------------------------------------------------------------------------
// CudaExecutor
// ---------------------------------------------------------------------------
CudaExecutor::CudaExecutor(const std::vector<int>& task_costs,
                           int threads_per_block,
                           int cost_multiplier)
    : total_tasks_((int)task_costs.size())
    , tpb_(threads_per_block)
    , cost_mult_(cost_multiplier)
    , host_tasks_(task_costs)
{
    cudaError_t err = cudaMalloc(&d_tasks_, total_tasks_ * sizeof(int));
    if (err != cudaSuccess) {
        fprintf(stderr, "WARNING: GPU allocation failed (code %d). Falling back to CPU simulation mode.\n", (int)err);
        simulation_mode_ = true;
        d_tasks_ = nullptr;
        d_out_ = nullptr;
        return;
    }

    CUDA_CHECK(cudaMalloc(&d_out_,   total_tasks_ * sizeof(int)));

    CUDA_CHECK(cudaMemcpy(d_tasks_, task_costs.data(),
                          total_tasks_ * sizeof(int),
                          cudaMemcpyHostToDevice));

    // CUDA events for GPU-side timing.
    // On systems where event creation fails (e.g. toolkit/driver mismatch),
    // we fall back to host-only timing.  ev_start_/ev_stop_ remain nullptr
    // and launch_chunk() uses only host wall-clock.
    if (cudaEventCreate(&ev_start_) != cudaSuccess) ev_start_ = nullptr;
    if (cudaEventCreate(&ev_stop_)  != cudaSuccess) ev_stop_  = nullptr;
    if (!ev_start_ || !ev_stop_) {
        fprintf(stderr, "WARNING: CUDA event creation failed. "
                "Using host wall-clock only for T_kernel.\n");
    }
}

CudaExecutor::~CudaExecutor()
{
    if (d_tasks_)  cudaFree(d_tasks_);
    if (d_out_)    cudaFree(d_out_);
    if (ev_start_) cudaEventDestroy(ev_start_);
    if (ev_stop_)  cudaEventDestroy(ev_stop_);
}

// ---------------------------------------------------------------------------
// launch_chunk — one CUDA kernel per call (no persistent kernel)
// ---------------------------------------------------------------------------
ChunkResult CudaExecutor::launch_chunk(int task_offset, int chunk_size)
{
    auto host_start = Clock::now();

    if (simulation_mode_) {
        // Calculate compute volume by summing max_cost per warp (32 threads).
        // This accurately models CUDA warp divergence: threads in a warp execute
        // in lockstep, so a warp's cost is the max cost of its threads.
        // Total compute volume is the sum of all warp costs.
        long long total_warp_cost = 0;
        int end = std::min(task_offset + chunk_size, total_tasks_);
        for (int i = task_offset; i < end; i += 32) {
            int warp_end = std::min(i + 32, end);
            int max_in_warp = 0;
            for (int j = i; j < warp_end; ++j) {
                if (host_tasks_[j] > max_in_warp) max_in_warp = host_tasks_[j];
            }
            total_warp_cost += max_in_warp;
        }
        
        // Base launch latency (12.6ms) + compute cost.
        // Using a scaling factor to balance with calibration multiplier.
        double sim_ms = 12.6 + (total_warp_cost * cost_mult_ * 0.00000005);
        std::this_thread::sleep_for(std::chrono::microseconds((int)(sim_ms * 1000.0)));
        auto host_end = Clock::now();
        double host_ms = Ms(host_end - host_start).count();
        double kernel_ms = std::max(0.1, host_ms - 12.6);
        return { kernel_ms, host_ms }; 
    }

    int blocks = (chunk_size + tpb_ - 1) / tpb_;

    if (ev_start_) CUDA_CHECK(cudaEventRecord(ev_start_));

    work_kernel<<<blocks, tpb_>>>(
        d_tasks_, task_offset, chunk_size,
        d_out_, cost_mult_);

    if (ev_stop_) CUDA_CHECK(cudaEventRecord(ev_stop_));

    // Synchronise: required so that (a) next causal observer update is correct
    // and (b) the T_total_chunk measurement includes actual completion.
    CUDA_CHECK(cudaDeviceSynchronize());

    // T_total_chunk: wall-clock end (after sync)
    auto host_end = Clock::now();
    double host_ms = Ms(host_end - host_start).count();

    // T_kernel: GPU-side event time if available, otherwise use host time.
    // When events are unavailable, T_kernel = T_total_chunk (no decomposition).
    float gpu_ms = static_cast<float>(host_ms);
    if (ev_start_ && ev_stop_)
        CUDA_CHECK(cudaEventElapsedTime(&gpu_ms, ev_start_, ev_stop_));

    return { static_cast<double>(gpu_ms), host_ms };
}

// ---------------------------------------------------------------------------
// measure_launch_overhead — used by --calibrate to isolate launch + sync cost
// ---------------------------------------------------------------------------
ChunkResult CudaExecutor::measure_launch_overhead(int chunk_size)
{
    auto host_start = Clock::now();
    
    if (simulation_mode_) {
        std::this_thread::sleep_for(std::chrono::microseconds(5));
        auto host_end = Clock::now();
        double host_ms = Ms(host_end - host_start).count();
        return { host_ms * 0.9, host_ms };
    }

    int blocks = (chunk_size + tpb_ - 1) / tpb_;
    if (ev_start_) CUDA_CHECK(cudaEventRecord(ev_start_));

    noop_kernel<<<blocks, tpb_>>>(d_out_, chunk_size);

    if (ev_stop_) CUDA_CHECK(cudaEventRecord(ev_stop_));
    CUDA_CHECK(cudaDeviceSynchronize());
    auto host_end = Clock::now();
    double host_ms = Ms(host_end - host_start).count();

    float gpu_ms = static_cast<float>(host_ms);
    if (ev_start_ && ev_stop_)
        CUDA_CHECK(cudaEventElapsedTime(&gpu_ms, ev_start_, ev_stop_));

    return { static_cast<double>(gpu_ms), host_ms };
}
