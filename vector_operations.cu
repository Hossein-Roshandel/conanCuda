/**
 * @file vector_operations.cu
 * @brief Implementation of CUDA vector operations
 */

#include "vector_operations.cuh"

__global__ void vectorAdd(const float* A, const float* B, float* C, int numElements) {
    // Calculate global thread ID: block position * block size + thread position within block
    int i = blockDim.x * blockIdx.x + threadIdx.x;

    // Bounds check: only process if within valid range
    if (i < numElements) {
        C[i] = A[i] + B[i];
    }
}
