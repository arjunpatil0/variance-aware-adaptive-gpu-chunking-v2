#include <cuda.h>
#include <cuda_runtime.h>
#include <stdio.h>
#include <vector>
#include <chrono>
#include <fstream>
#include <string>

// ------------------------------------------------------------
// Same irregular kernel as adaptive version
// ------------------------------------------------------------
__global__ void work_kernel(int *tasks, int start, int chunk, int *out) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int idx = start + tid;

    if (tid < chunk) {
        int work = tasks[idx];
        int acc = 0;

        for (int i = 0; i < work; i++) {
            acc += (i ^ work) & 0xFF;
        }

        out[idx] = acc;
    }
}

int main(int argc, char *argv[]) {

    // ------------------------------------------------------------
    // Trial number
    // Usage: static_chunking.exe 1
    // ------------------------------------------------------------
    int trial = 1;

    if (argc > 1) {
        trial = std::stoi(argv[1]);
    }

    // ------------------------------------------------------------
    // Configuration (FIXED chunk size)
    // ------------------------------------------------------------
    const int NUM_TASKS = 5'000'000;
    const int KERNEL_REPEATS = 1000;
    const int FIXED_CHUNK = 100'000;

    // ------------------------------------------------------------
    // Generate identical irregular workload
    // ------------------------------------------------------------
    std::vector<int> h_tasks(NUM_TASKS);

    for (int i = 0; i < NUM_TASKS; i++) {
        h_tasks[i] = 500 + (i % 5000);
    }

    // ------------------------------------------------------------
    // GPU memory
    // ------------------------------------------------------------
    int *d_tasks, *d_out;

    cudaMalloc(&d_tasks, NUM_TASKS * sizeof(int));
    cudaMalloc(&d_out, NUM_TASKS * sizeof(int));

    cudaMemcpy(
        d_tasks,
        h_tasks.data(),
        NUM_TASKS * sizeof(int),
        cudaMemcpyHostToDevice
    );

    // ------------------------------------------------------------
    // Unique CSV for each trial
    // ------------------------------------------------------------
    std::string filename =
        "../data/static_trial_" +
        std::to_string(trial) +
        ".csv";

    std::ofstream logfile(filename);

    logfile << "chunk_id,chunk_size,time_ms\n";

    printf("Running Static Trial %d\n", trial);
    printf("Saving results to: %s\n", filename.c_str());

    // ------------------------------------------------------------
    // Static scheduling loop
    // ------------------------------------------------------------
    int processed = 0;
    int chunk_id = 0;

    while (processed < NUM_TASKS) {

        chunk_id++;

        int current_chunk =
            (processed + FIXED_CHUNK > NUM_TASKS)
            ? (NUM_TASKS - processed)
            : FIXED_CHUNK;

        int threads = 256;
        int blocks =
            (current_chunk + threads - 1) / threads;

        auto t1 =
            std::chrono::high_resolution_clock::now();

        for (int r = 0; r < KERNEL_REPEATS; r++) {

            work_kernel<<<blocks, threads>>>(
                d_tasks,
                processed,
                current_chunk,
                d_out
            );
        }

        cudaDeviceSynchronize();

        auto t2 =
            std::chrono::high_resolution_clock::now();

        double ms =
            std::chrono::duration<double, std::milli>(
                t2 - t1
            ).count();

        printf(
            "Chunk %d | size %d | time %.2f ms\n",
            chunk_id,
            current_chunk,
            ms
        );

        logfile
            << chunk_id << ","
            << current_chunk << ","
            << ms << "\n";

        processed += current_chunk;
    }

    logfile.close();

    cudaFree(d_tasks);
    cudaFree(d_out);

    return 0;
}