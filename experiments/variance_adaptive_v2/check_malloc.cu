#include <cuda_runtime.h>
#include <stdio.h>

__global__ void dummy() {}

int main() {
    void* ptr = NULL;
    cudaError_t err = cudaMalloc(&ptr, 4);
    printf("err: %d ptr: %p\n", (int)err, ptr);
    return 0;
}
