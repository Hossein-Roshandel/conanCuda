cmake_minimum_required(VERSION 3.18.0)  # CUDA requires CMake 3.18+
project(conancuda VERSION 0.1.0 LANGUAGES C CXX CUDA)

# Set C++ standard
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Set CUDA standard
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

# Find packages
find_package(cuda-api-wrappers REQUIRED)

# Add executables
add_executable(conancuda main.cpp)
add_executable(vectoradd vectorAdd.cu)

# Set CUDA architectures (adjust based on your target GPUs)
set_property(TARGET vectoradd PROPERTY CUDA_ARCHITECTURES 75 80 86)

# Link libraries
target_link_libraries(vectoradd 
    PRIVATE 
        cuda-api-wrappers::runtime-and-driver 
        cuda
)

# Set compiler-specific options
if(CMAKE_CUDA_COMPILER_ID STREQUAL "NVIDIA")
    target_compile_options(vectoradd PRIVATE 
        $<$<COMPILE_LANGUAGE:CUDA>:
            --extended-lambda
            --expt-relaxed-constexpr
        >
    )
endif()

# Enable testing
include(CTest)
enable_testing()

# Add simple tests
add_test(NAME test_conancuda COMMAND conancuda)
add_test(NAME test_vectoradd COMMAND vectoradd)

# CPack configuration
set(CPACK_PROJECT_NAME ${PROJECT_NAME})
set(CPACK_PROJECT_VERSION ${PROJECT_VERSION})
include(CPack)

# Print configuration information
message(STATUS "Project: ${PROJECT_NAME} ${PROJECT_VERSION}")
message(STATUS "CUDA Compiler: ${CMAKE_CUDA_COMPILER}")
message(STATUS "CUDA Version: ${CMAKE_CUDA_COMPILER_VERSION}")
message(STATUS "C++ Standard: ${CMAKE_CXX_STANDARD}")
message(STATUS "CUDA Standard: ${CMAKE_CUDA_STANDARD}")