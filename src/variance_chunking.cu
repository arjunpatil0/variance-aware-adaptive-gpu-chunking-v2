#include <cuda.h>
#include <cuda_runtime.h>
#include <stdio.h>
#include <vector>
#include <chrono>
#include <fstream>
#include <cmath>
#include <string>

// ------------------------------------------------------------
// Irregular GPU kernel
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

    int trial = 1;
    if (argc > 1)
        trial = std::stoi(argv[1]);

    // ------------------------------------------------------------
    // Configuration
    // ------------------------------------------------------------
    const int NUM_TASKS = 5'000'000;
    const int KERNEL_REPEATS = 1000;

    const int WINDOW = 5;

    // Less sensitive thresholds
    const double VAR_HIGH = 0.80;
    const double VAR_LOW  = 0.10;

    // Start from the same chunk size as static
    int chunk = 100'000;
    const int min_chunk = 25'000;
    const int max_chunk = 200'000;
    const int chunk_step = 10'000;

    // Wait before making another adjustment
    const int COOLDOWN = 3;
    int cooldown_remaining = 0;

    // ------------------------------------------------------------
    // Generate irregular workload
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
        "../data/adaptive_tuned_trial_" +
        std::to_string(trial) +
        ".csv";

    std::ofstream logfile(filename);
    logfile << "chunk_id,chunk_size,time_ms,variance\n";

    printf("Running Tuned Adaptive Trial %d\n", trial);
    printf("Saving results to: %s\n", filename.c_str());

    // ------------------------------------------------------------
    // Scheduling state
    // ------------------------------------------------------------
    std::vector<double> window;
    int processed = 0;
    int chunk_id = 0;

    // ------------------------------------------------------------
    // Adaptive scheduling loop
    // ------------------------------------------------------------
    while (processed < NUM_TASKS) {

        chunk_id++;

        int current_chunk =
            (processed + chunk > NUM_TASKS)
            ? (NUM_TASKS - processed)
            : chunk;

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

        // --------------------------------------------------------
        // Update sliding window
        // --------------------------------------------------------
        window.push_back(ms);

        if (window.size() > WINDOW)
            window.erase(window.begin());

        double variance = 0.0;

        if (window.size() == WINDOW) {

            double mean = 0.0;

            for (double t : window)
                mean += t;

            mean /= WINDOW;

            for (double t : window)
                variance += (t - mean) * (t - mean);

            variance /= WINDOW;

            // ----------------------------------------------------
            // Variance-aware control with cooldown
            // ----------------------------------------------------
            if (cooldown_remaining > 0) {

                cooldown_remaining--;

            } else if (variance > VAR_HIGH &&
                       chunk > min_chunk) {

                // Reduce gradually: 100k -> 75k -> 56.25k ...
                chunk = (chunk * 3) / 4;

                if (chunk < min_chunk)
                    chunk = min_chunk;

                cooldown_remaining = COOLDOWN;

            } else if (variance < VAR_LOW &&
                       chunk < max_chunk) {

                chunk += chunk_step;

                if (chunk > max_chunk)
                    chunk = max_chunk;

                cooldown_remaining = COOLDOWN;
            }
        }

        printf(
            "Chunk %d | size %d | time %.2f ms | var %.3f\n",
            chunk_id,
            current_chunk,
            ms,
            variance
        );

        logfile
            << chunk_id << ","
            << current_chunk << ","
            << ms << ","
            << variance << "\n";

        processed += current_chunk;
    }

    logfile.close();

    cudaFree(d_tasks);
    cudaFree(d_out);

    return 0;
}