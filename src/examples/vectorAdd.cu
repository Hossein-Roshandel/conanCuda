/**
 * @file vectorAdd.cu
 * @brief CUDA Vector Addition Example using cuda-api-wrappers
 *
 * This example demonstrates how the cuda-api-wrappers library provides a modern C++
 * interface for CUDA programming, abstracting away many of the low-level CUDA runtime
 * API calls while maintaining performance and expressiveness.
 *
 * Derived from the nVIDIA CUDA 8.0 samples by Eyal Rozenberg
 *
 * The derivation is specifically permitted in the nVIDIA CUDA Samples EULA
 * and derived by the owner of this code according to the EULA.
 *
 * @author Eyal Rozenberg
 * @see https://github.com/eyalroz/cuda-api-wrappers
 *
 * Use this reasonably. If you want to discuss licensing formalities, please
 * contact the author.
 */

#include <cuda/api.hpp>  // Modern C++ wrapper for CUDA Runtime API

#include <algorithm>
#include <iostream>
#include <memory>
#include <random>
#include <vector>

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
__global__ void vectorAdd(const float* A, const float* B, float* C, int numElements) {
    // Calculate global thread ID: block position * block size + thread position within block
    int i = blockDim.x * blockIdx.x + threadIdx.x;

    // Bounds check: only process if within valid range
    if (i < numElements) {
        C[i] = A[i] + B[i];
    }
}

/**
 * @brief Main function demonstrating CUDA vector addition with cuda-api-wrappers
 *
 * This program demonstrates several key features of the cuda-api-wrappers library:
 * 1. Device query and selection using cuda::device API
 * 2. Automatic memory management with smart pointers (make_unique_span)
 * 3. Simplified memory transfer operations
 * 4. High-level kernel launch configuration
 * 5. RAII-based resource management (automatic cleanup)
 *
 * @return EXIT_SUCCESS if test passes, EXIT_FAILURE otherwise
 */
int main() {
    // Check for available CUDA devices using the wrapper's device API
    // This replaces cudaGetDeviceCount() with a more intuitive interface
    if (cuda::device::count() == 0) {
        std::cerr << "No CUDA devices on this system" << "\n";
        exit(EXIT_FAILURE);
    }

    int numElements = 50000;
    std::cout << "[Vector addition of " << numElements << " elements]\n";

    // Allocate host memory using STL containers
    // std::vector provides automatic memory management and exception safety
    auto h_A = std::vector<float>(numElements);
    auto h_B = std::vector<float>(numElements);
    auto h_C = std::vector<float>(numElements);

    // Initialize input vectors with random values in [0.0, 1.0]
    // Lambda function encapsulates random number generation state
    auto generator = []() {
        static std::random_device random_device;
        static std::mt19937 randomness_generator{random_device()};
        static std::uniform_real_distribution<float> distribution{0.0, 1.0};
        return distribution(randomness_generator);
    };
    std::generate(h_A.begin(), h_A.end(), generator);
    std::generate(h_B.begin(), h_B.end(), generator);

    // Get current CUDA device - wrapper handles device context
    // This replaces cudaGetDevice() and provides a typed device object
    auto device = cuda::device::current::get();

    // Allocate device memory using smart pointers - automatic cleanup on scope exit!
    // make_unique_span returns a unique_ptr-like object that automatically frees memory
    // This replaces cudaMalloc() and eliminates manual cudaFree() calls
    auto d_A = cuda::memory::make_unique_span<float>(device, numElements);
    auto d_B = cuda::memory::make_unique_span<float>(device, numElements);
    auto d_C = cuda::memory::make_unique_span<float>(device, numElements);

    // Copy data from host to device - wrapper handles direction automatically
    // This replaces cudaMemcpy(d_A, h_A.data(), size, cudaMemcpyHostToDevice)
    cuda::memory::copy(d_A, h_A);
    cuda::memory::copy(d_B, h_B);

    // Configure kernel launch parameters using builder pattern
    // The wrapper calculates optimal grid dimensions automatically
    // overall_size: total number of elements to process
    // block_size: threads per block (must be multiple of warp size, typically 32)
    auto launch_config =
        cuda::launch_config_builder().overall_size(numElements).block_size(256).build();

    std::cout << "CUDA kernel launch with " << launch_config.dimensions.grid.x << " blocks of "
              << launch_config.dimensions.block.x << " threads each\n";

    // Launch kernel with high-level API - replaces the <<<>>> syntax
    // The wrapper handles dimension setup and error checking
    // Arguments: kernel function, launch config, kernel parameters
    cuda::launch(vectorAdd, launch_config, d_A.data(), d_B.data(), d_C.data(), numElements);

    // Copy results back from device to host
    // This replaces cudaMemcpy(h_C.data(), d_C, size, cudaMemcpyDeviceToHost)
    cuda::memory::copy(h_C, d_C);

    // Verify that the result vector is correct
    // Check each element with floating-point tolerance
    for (int i = 0; i < numElements; ++i) {
        if (fabs(h_A[i] + h_B[i] - h_C[i]) > 1e-5) {
            std::cerr << "Result verification failed at element " << i << "\n";
            exit(EXIT_FAILURE);
        }
    }

    std::cout << "Test PASSED\n";
    std::cout << "SUCCESS\n";

    // Note: No explicit cleanup needed!
    // The cuda-api-wrappers smart pointers automatically free device memory
    // This demonstrates RAII (Resource Acquisition Is Initialization) principle
}
