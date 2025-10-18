cmake_minimum_required(VERSION 3.18.0) # CUDA requires CMake 3.18+
project(
  conancuda
  VERSION 0.1.0
  LANGUAGES C CXX CUDA)

# Set C++ standard
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# Set CUDA standard
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

# Export compile commands for clang-tidy and other tools
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

# Options for optional features
option(BUILD_TESTS "Build unit tests" ON)
option(BUILD_DOCS "Build documentation with Doxygen" OFF)
option(ENABLE_CLANG_TIDY "Enable clang-tidy static analysis" OFF)

# Find packages
find_package(cuda-api-wrappers REQUIRED)

# Add shared vector operations library
add_library(vector_operations STATIC vector_operations.cu)
target_include_directories(vector_operations PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})
set_property(TARGET vector_operations PROPERTY CUDA_ARCHITECTURES 75 80 86)
target_link_libraries(vector_operations
                      PUBLIC cuda-api-wrappers::runtime-and-driver cuda)

# Add executables
add_executable(conancuda main.cpp)
add_executable(vectoradd vectorAdd.cu)

# Set CUDA architectures (adjust based on your target GPUs) 75: Turing, 80:
# Ampere, 86: Ampere (RTX 30xx)
set_property(TARGET vectoradd PROPERTY CUDA_ARCHITECTURES 75 80 86)

# Link libraries
target_link_libraries(vectoradd PRIVATE cuda-api-wrappers::runtime-and-driver
                                        cuda)

# Set compiler-specific options
if(CMAKE_CUDA_COMPILER_ID STREQUAL "NVIDIA")
  target_compile_options(
    vectoradd PRIVATE $<$<COMPILE_LANGUAGE:CUDA>: --extended-lambda
                      --expt-relaxed-constexpr >)
  target_compile_options(
    vector_operations PRIVATE $<$<COMPILE_LANGUAGE:CUDA>: --extended-lambda
                              --expt-relaxed-constexpr >)
endif()

# Compiler warnings
if(CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
  add_compile_options(-Wall -Wextra -Wpedantic)
endif()

# Enable testing
include(CTest)
enable_testing()

# Add simple tests for main executables
add_test(NAME test_conancuda COMMAND conancuda)
add_test(NAME test_vectoradd COMMAND vectoradd)

# Build unit tests if enabled
if(BUILD_TESTS)
  find_package(GTest REQUIRED)

  add_executable(test_vector_add tests/test_vector_add.cu)
  target_link_libraries(
    test_vector_add PRIVATE vector_operations GTest::gtest GTest::gtest_main
                            cuda-api-wrappers::runtime-and-driver cuda)
  set_property(TARGET test_vector_add PROPERTY CUDA_ARCHITECTURES 75 80 86)

  # Add to CTest
  add_test(NAME unit_tests COMMAND test_vector_add)

  # Set test properties
  set_tests_properties(unit_tests PROPERTIES LABELS "unit" TIMEOUT 300)
endif()

# Doxygen documentation
if(BUILD_DOCS)
  find_package(Doxygen)
  if(DOXYGEN_FOUND)
    set(DOXYGEN_IN ${CMAKE_CURRENT_SOURCE_DIR}/Doxyfile)
    set(DOXYGEN_OUT ${CMAKE_CURRENT_BINARY_DIR}/Doxyfile)

    configure_file(${DOXYGEN_IN} ${DOXYGEN_OUT} @ONLY)

    add_custom_target(
      docs
      COMMAND ${DOXYGEN_EXECUTABLE} ${DOXYGEN_OUT}
      WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
      COMMENT "Generating API documentation with Doxygen"
      VERBATIM)

    message(STATUS "Doxygen found. Run 'make docs' to generate documentation.")
  else()
    message(WARNING "Doxygen not found. Documentation will not be generated.")
  endif()
endif()

# Clang-tidy integration
if(ENABLE_CLANG_TIDY)
  find_program(CLANG_TIDY_EXE NAMES clang-tidy)
  if(CLANG_TIDY_EXE)
    set(CMAKE_CXX_CLANG_TIDY ${CLANG_TIDY_EXE})
    message(STATUS "clang-tidy found: ${CLANG_TIDY_EXE}")
  else()
    message(WARNING "clang-tidy not found")
  endif()
endif()

# Add custom targets for code quality
add_custom_target(
  format
  COMMAND find ${CMAKE_SOURCE_DIR} -name "*.cpp" -o -name "*.hpp" -o -name
          "*.cu" -o -name "*.cuh" | xargs clang-format -i --style=file
  COMMENT "Formatting code with clang-format"
  VERBATIM)

add_custom_target(
  static-analysis
  COMMAND ${CMAKE_CURRENT_SOURCE_DIR}/scripts/static_analysis.sh
  WORKING_DIRECTORY ${CMAKE_SOURCE_DIR}
  COMMENT "Running static analysis with clang-tidy"
  VERBATIM)

add_custom_target(
  memory-check
  COMMAND ${CMAKE_CURRENT_SOURCE_DIR}/scripts/memory_check.sh
          ${CMAKE_CURRENT_BINARY_DIR}/vectoradd
  WORKING_DIRECTORY ${CMAKE_SOURCE_DIR}
  COMMENT "Running memory checks with Valgrind and cuda-memcheck"
  VERBATIM)

# CPack configuration
set(CPACK_PROJECT_NAME ${PROJECT_NAME})
set(CPACK_PROJECT_VERSION ${PROJECT_VERSION})
include(CPack)

# Print configuration information
message(STATUS "========================================")
message(STATUS "Configuration Summary")
message(STATUS "========================================")
message(STATUS "Project: ${PROJECT_NAME} ${PROJECT_VERSION}")
message(STATUS "CUDA Compiler: ${CMAKE_CUDA_COMPILER}")
message(STATUS "CUDA Version: ${CMAKE_CUDA_COMPILER_VERSION}")
message(STATUS "C++ Standard: ${CMAKE_CXX_STANDARD}")
message(STATUS "CUDA Standard: ${CMAKE_CUDA_STANDARD}")
message(STATUS "Build Type: ${CMAKE_BUILD_TYPE}")
message(STATUS "Build Tests: ${BUILD_TESTS}")
message(STATUS "Build Docs: ${BUILD_DOCS}")
message(STATUS "Enable Clang-Tidy: ${ENABLE_CLANG_TIDY}")
message(STATUS "========================================")
