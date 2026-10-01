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
    printf("Starting test2...\n");
    int *d_tasks;
    cudaError_t err = cudaMalloc(&d_tasks, 4);
    printf("err: %d\n", (int)err);
    return 0;
}