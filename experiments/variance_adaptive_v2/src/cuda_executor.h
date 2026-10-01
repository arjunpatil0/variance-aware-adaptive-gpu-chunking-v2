#pragma once
// cuda_executor.h — CUDA kernel launch wrapper with CUDA event timing.
//
// WHY a wrapper class: encapsulating d_tasks/d_out allocation and the CUDA
// event lifecycle here means the scheduler code and the experiment runner
// never touch raw CUDA APIs, reducing the chance of resource-management bugs.
//
// Timing separation:
//   T_kernel  = CUDA event elapsed time (GPU-side only)
//   T_total_chunk = chrono wall-clock including launch latency and sync wait
// Both are recorded per chunk.  See scheduler.h::SchedulerDecision.
//
// One kernel launch per call to launch_chunk().
// There is no persistent kernel.  This is required by the experimental design:
// the number of kernel launches must equal the number of scheduler decisions.

#include "config.h"
#include <vector>
#include <cuda_runtime.h>

// ---------------------------------------------------------------------------
// CUDA error-check macro
// Prints the error and exits; no silent recovery.
// ---------------------------------------------------------------------------
#define CUDA_CHECK(call)                                                    \
    do {                                                                    \
        cudaError_t _err = (call);                                          \
        if (_err != cudaSuccess) {                                          \
            const char* _msg = cudaGetErrorString(_err);                    \
            fprintf(stderr, "CUDA error %s:%d  code=%d  %s\n",             \
                    __FILE__, __LINE__, (int)_err,                          \
                    _msg ? _msg : "(null)");                                \
            exit(EXIT_FAILURE);                                             \
        }                                                                   \
    } while (0)

// ---------------------------------------------------------------------------
// Result from a single chunk launch
// ---------------------------------------------------------------------------
struct ChunkResult {
    double t_kernel_ms;        // CUDA event elapsed time (GPU execution only)
    double t_total_chunk_ms;   // host wall-clock: launch + GPU + sync
};

// ---------------------------------------------------------------------------
// CudaExecutor
// ---------------------------------------------------------------------------
class CudaExecutor {
public:
    // Uploads task_costs to the GPU.  Allocates d_out.
    // cost_multiplier is forwarded to the kernel (W *= cost_multiplier).
    CudaExecutor(const std::vector<int>& task_costs,
                 int threads_per_block,
                 int cost_multiplier);
    ~CudaExecutor();

    // Disallow copy (manages raw GPU memory).
    CudaExecutor(const CudaExecutor&)            = delete;
    CudaExecutor& operator=(const CudaExecutor&) = delete;

    // Launch exactly ONE kernel for tasks [task_offset, task_offset+chunk_size).
    // Synchronizes before returning.
    ChunkResult launch_chunk(int task_offset, int chunk_size);

    // Launch a zero-work kernel of the given chunk_size to measure
    // launch + sync overhead.  Used during --calibrate mode.
    ChunkResult measure_launch_overhead(int chunk_size);

    int total_tasks()      const { return total_tasks_; }
    int threads_per_block() const { return tpb_; }

private:
    int*   d_tasks_       = nullptr;
    int*   d_out_         = nullptr;
    int    total_tasks_   = 0;
    int    tpb_           = 256;
    int    cost_mult_     = 1;

    // Pre-allocated CUDA events reused every launch to avoid event-creation
    // overhead inside the hot path.
    cudaEvent_t ev_start_ = nullptr;
    cudaEvent_t ev_stop_  = nullptr;
    bool simulation_mode_ = false;
    std::vector<int> host_tasks_;
};
