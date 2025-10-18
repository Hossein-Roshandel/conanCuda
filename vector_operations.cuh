/**
 * @file vector_operations.cuh
 * @brief CUDA vector operations header
 *
 * This header declares CUDA kernels for vector operations
 */

#ifndef VECTOR_OPERATIONS_CUH
#define VECTOR_OPERATIONS_CUH

/**
 * @brief CUDA kernel for element-wise vector addition
 *
 * This kernel performs parallel addition of two vectors A and B, storing the result in C.
 * Each thread computes one element of the output vector using the formula: C[i] = A[i] + B[i]
 *
 * @param A Input vector A (device memory)
 * @param B Input vector B (device memory)
 * @param C Output vector C (device memory)
 * @param numElements Number of elements in each vector
 *
 * @note Thread safety: Each thread operates on a unique element, so no synchronization is needed
 * @note The bounds check (i < numElements) handles cases where grid size isn't a perfect multiple
 */
__global__ void vectorAdd(const float* A, const float* B, float* C, int numElements);

#endif  // VECTOR_OPERATIONS_CUH
