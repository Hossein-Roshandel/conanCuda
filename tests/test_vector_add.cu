/**
 * @file test_vector_add.cu
 * @brief Unit tests for CUDA vector addition kernel
 *
 * This file contains comprehensive unit tests for the vectorAdd kernel
 * using Google Test framework. Tests cover various scenarios including
 * edge cases, different data sizes, and numerical accuracy.
 */

#include <cuda/api.hpp>

#include <cmath>
#include <gtest/gtest.h>
#include <numeric>
#include <random>
#include <vector>

#include "vector_operations.cuh"

// Define a GTest fixture for CUDA-related tests
class CudaFixture : public ::testing::Test {
   protected:
    float* d_data = nullptr;                // Pointer to device memory
    size_t data_size = 10 * sizeof(float);  // Size of data to allocate

    void SetUp() override {
        // Initialize CUDA device
        int deviceCount;
        cudaGetDeviceCount(&deviceCount);
        ASSERT_GT(deviceCount, 0) << "No CUDA devices found.";
        cudaSetDevice(0);  // Use device 0

        // Allocate memory on the device
        cudaError_t err = cudaMalloc(&d_data, data_size);
        ASSERT_EQ(err, cudaSuccess) << "cudaMalloc failed: " << cudaGetErrorString(err);

        // Optionally, initialize device memory with some values
        float* h_data = new float[data_size / sizeof(float)];
        for (size_t i = 0; i < data_size / sizeof(float); ++i) {
            h_data[i] = i;
        }
        err = cudaMemcpy(d_data, h_data, data_size, cudaMemcpyHostToDevice);
        ASSERT_EQ(err, cudaSuccess) << "cudaMemcpy failed: " << cudaGetErrorString(err);
        delete[] h_data;
    }

    void TearDown() override {
        // Free device memory
        if (d_data) {
            cudaFree(d_data);
            d_data = nullptr;
        }
        // Reset CUDA device (optional, but good practice for clean state)
        cudaDeviceReset();
    }
};

// Test case using the CudaFixture
TEST_F(CudaFixture, SimpleKernelExecution) {
    float* d_output = nullptr;
    cudaError_t err = cudaMalloc(&d_output, data_size);
    ASSERT_EQ(err, cudaSuccess) << "cudaMalloc for output failed: " << cudaGetErrorString(err);

    int numElements = data_size / sizeof(int);
    vectorAdd<<<1, numElements>>>(d_data, d_data, d_output, numElements);
    err = cudaGetLastError();
    ASSERT_EQ(err, cudaSuccess) << "Kernel launch failed: " << cudaGetErrorString(err);

    // Copy results back to host for verification
    float* h_output = new float[numElements];
    err = cudaMemcpy(h_output, d_output, data_size, cudaMemcpyDeviceToHost);
    ASSERT_EQ(err, cudaSuccess) << "cudaMemcpy back to host failed: " << cudaGetErrorString(err);

    // Verify results
    for (int i = 0; i < numElements; ++i) {
        EXPECT_EQ(h_output[i], i * 2) << "Mismatch at index " << i;
    }

    delete[] h_output;
    cudaFree(d_output);
}

// /**
//  * @class VectorAddTest
//  * @brief Test fixture for vector addition tests
//  *
//  * Provides common setup and utility functions for vector addition tests
//  */
// class VectorAddTest : public ::testing::Test {
//    protected:
//     void SetUp() override {
//         // Check for CUDA devices
//         if (cuda::device::count() == 0) {
//             GTEST_SKIP() << "No CUDA devices available";
//         }
//         device = cuda::device::current::get();
//     }

//     /**
//      * @brief Helper function to run vector addition and verify results
//      *
//      * @param numElements Number of elements in vectors
//      * @param initA Function to initialize vector A
//      * @param initB Function to initialize vector B
//      * @param tolerance Floating-point comparison tolerance
//      */
//     void runVectorAddTest(int numElements, std::function<void(std::vector<float>&)> initA,
//                           std::function<void(std::vector<float>&)> initB, float tolerance = 1e-5)
//                           {
//         // Allocate and initialize host vectors
//         std::vector<float> h_A(numElements);
//         std::vector<float> h_B(numElements);
//         std::vector<float> h_C(numElements);

//         initA(h_A);
//         initB(h_B);

//         // Allocate device memory
//         auto d_A = cuda::memory::make_unique_span<float>(device, numElements);
//         auto d_B = cuda::memory::make_unique_span<float>(device, numElements);
//         auto d_C = cuda::memory::make_unique_span<float>(device, numElements);

//         // Copy to device
//         cuda::memory::copy(d_A, h_A);
//         cuda::memory::copy(d_B, h_B);

//         // Launch kernel
//         auto launch_config =
//             cuda::launch_config_builder().overall_size(numElements).block_size(256).build();

//         cuda::launch(vectorAdd, launch_config, d_A.data(), d_B.data(), d_C.data(), numElements);

//         // Copy results back
//         cuda::memory::copy(h_C, d_C);

//         // Verify results
//         for (int i = 0; i < numElements; ++i) {
//             EXPECT_NEAR(h_A[i] + h_B[i], h_C[i], tolerance) << "Mismatch at index " << i;
//         }
//     }

//     cuda::device_t device;
// };

// /**
//  * @test Basic vector addition with small size
//  */
// TEST_F(VectorAddTest, SmallVector) {
//     const int numElements = 1024;

//     auto initRandom = [](std::vector<float>& vec) {
//         std::random_device rd;
//         std::mt19937 gen(rd());
//         std::uniform_real_distribution<float> dis(0.0f, 1.0f);
//         for (auto& val : vec) {
//             val = dis(gen);
//         }
//     };

//     runVectorAddTest(numElements, initRandom, initRandom);
// }

// /**
//  * @test Large vector addition
//  */
// TEST_F(VectorAddTest, LargeVector) {
//     const int numElements = 1000000;

//     auto initRandom = [](std::vector<float>& vec) {
//         std::random_device rd;
//         std::mt19937 gen(rd());
//         std::uniform_real_distribution<float> dis(-10.0f, 10.0f);
//         for (auto& val : vec) {
//             val = dis(gen);
//         }
//     };

//     runVectorAddTest(numElements, initRandom, initRandom, 1e-4);
// }

// /**
//  * @test Vector addition with zeros
//  */
// TEST_F(VectorAddTest, ZeroVectors) {
//     const int numElements = 5000;

//     auto initZeros = [](std::vector<float>& vec) { std::fill(vec.begin(), vec.end(), 0.0f); };

//     auto initRandom = [](std::vector<float>& vec) {
//         std::random_device rd;
//         std::mt19937 gen(rd());
//         std::uniform_real_distribution<float> dis(0.0f, 1.0f);
//         for (auto& val : vec) {
//             val = dis(gen);
//         }
//     };

//     // Test A + 0 = A
//     runVectorAddTest(numElements, initRandom, initZeros);
// }

// /**
//  * @test Vector addition with ones
//  */
// TEST_F(VectorAddTest, OnesVector) {
//     const int numElements = 10000;

//     auto initOnes = [](std::vector<float>& vec) { std::fill(vec.begin(), vec.end(), 1.0f); };

//     runVectorAddTest(numElements, initOnes, initOnes);
// }

// /**
//  * @test Vector addition with negative numbers
//  */
// TEST_F(VectorAddTest, NegativeNumbers) {
//     const int numElements = 5000;

//     auto initNegative = [](std::vector<float>& vec) {
//         std::random_device rd;
//         std::mt19937 gen(rd());
//         std::uniform_real_distribution<float> dis(-100.0f, -1.0f);
//         for (auto& val : vec) {
//             val = dis(gen);
//         }
//     };

//     auto initPositive = [](std::vector<float>& vec) {
//         std::random_device rd;
//         std::mt19937 gen(rd());
//         std::uniform_real_distribution<float> dis(1.0f, 100.0f);
//         for (auto& val : vec) {
//             val = dis(gen);
//         }
//     };

//     runVectorAddTest(numElements, initNegative, initPositive, 1e-4);
// }

// /**
//  * @test Edge case: Single element
//  */
// TEST_F(VectorAddTest, SingleElement) {
//     const int numElements = 1;

//     auto initValue = [](std::vector<float>& vec) { vec[0] = 42.0f; };

//     runVectorAddTest(numElements, initValue, initValue);
// }

// /**
//  * @test Non-aligned size (not a multiple of block size)
//  */
// TEST_F(VectorAddTest, NonAlignedSize) {
//     const int numElements = 50001;  // Not a multiple of 256

//     auto initRandom = [](std::vector<float>& vec) {
//         std::random_device rd;
//         std::mt19937 gen(rd());
//         std::uniform_real_distribution<float> dis(0.0f, 1.0f);
//         for (auto& val : vec) {
//             val = dis(gen);
//         }
//     };

//     runVectorAddTest(numElements, initRandom, initRandom);
// }

// /**
//  * @test Performance benchmark test (not a failure test)
//  */
// TEST_F(VectorAddTest, PerformanceBenchmark) {
//     const int numElements = 10000000;  // 10M elements

//     std::vector<float> h_A(numElements, 1.0f);
//     std::vector<float> h_B(numElements, 2.0f);
//     std::vector<float> h_C(numElements);

//     auto d_A = cuda::memory::make_unique_span<float>(device, numElements);
//     auto d_B = cuda::memory::make_unique_span<float>(device, numElements);
//     auto d_C = cuda::memory::make_unique_span<float>(device, numElements);

//     cuda::memory::copy(d_A, h_A);
//     cuda::memory::copy(d_B, h_B);

//     auto launch_config =
//         cuda::launch_config_builder().overall_size(numElements).block_size(256).build();

//     // Warm-up run
//     cuda::launch(vectorAdd, launch_config, d_A.data(), d_B.data(), d_C.data(), numElements);
//     cuda::device::current::get().synchronize();

//     // Timed run
//     auto start = std::chrono::high_resolution_clock::now();
//     cuda::launch(vectorAdd, launch_config, d_A.data(), d_B.data(), d_C.data(), numElements);
//     cuda::device::current::get().synchronize();
//     auto end = std::chrono::high_resolution_clock::now();

//     auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start);

//     std::cout << "Performance: " << numElements << " elements in " << duration.count()
//               << " microseconds" << std::endl;
//     std::cout << "Throughput: " << (numElements * sizeof(float) * 3 / duration.count()) << "
//     MB/s"
//               << std::endl;

//     // Just verify correctness, not performance
//     cuda::memory::copy(h_C, d_C);
//     EXPECT_FLOAT_EQ(h_C[0], 3.0f);
//     EXPECT_FLOAT_EQ(h_C[numElements - 1], 3.0f);
// }

/**
 * @brief Main test runner
 */
int main(int argc, char** argv) {
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
