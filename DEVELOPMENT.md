# Development Guide

This guide covers advanced development topics, best practices, and detailed information about the
tools and workflows in this project.

## Table of Contents

- [Development Environment Setup](#development-environment-setup)
- [Code Style and Formatting](#code-style-and-formatting)
- [Testing Strategy](#testing-strategy)
- [Static Analysis](#static-analysis)
- [Memory Analysis](#memory-analysis)
- [Documentation](#documentation)
- [Debugging](#debugging)
- [Performance Optimization](#performance-optimization)
- [CI/CD Integration](#cicd-integration)

## Project Structure

This project follows a standard C++ directory layout:

- **src/examples/**: Example applications demonstrating CUDA features
- **src/kernels/**: Reusable CUDA kernel implementations
- **include/**: Public header files for the project
- **tests/**: Unit tests using Google Test
- **scripts/**: Build and analysis utility scripts
- **docs/**: Generated documentation (Doxygen output)

See [REORGANIZATION.md](REORGANIZATION.md) for detailed information about the structure.

## Development Environment Setup

### Local Development

1. **Install Required Tools**:

   ```bash
   # Ubuntu/Debian
   sudo apt update
   sudo apt install -y \
       cmake \
       ninja-build \
       nvidia-cuda-toolkit \
       clang-format \
       clang-tidy \
       doxygen \
       graphviz \
       valgrind \
       python3-pip

   # Install uv
   curl -LsSf https://astral.sh/uv/install.sh | sh

   # Install pre-commit
   pip install pre-commit
   ```

1. **Configure Git Hooks**:

   ```bash
   pre-commit install
   pre-commit install --hook-type commit-msg
   ```

1. **Verify Setup**:

   ```bash
   make info
   ```

### Dev Container Development

The dev container provides a complete, pre-configured environment with all tools installed.

**Advantages**:

- Consistent environment across team members
- No local CUDA installation required for development
- All tools pre-installed and configured
- Isolated from host system

**To use**:

1. Install Docker and VS Code with Dev Containers extension
1. Open project in VS Code
1. Click "Reopen in Container" when prompted

## Code Style and Formatting

### C++ and CUDA Code

We use **clang-format** with a configuration based on Google style with CUDA-specific adaptations.

**Key style guidelines**:

- **Indentation**: 4 spaces (no tabs)
- **Column Limit**: 100 characters
- **Braces**: Attached style (`if (condition) {`)
- **Naming**:
  - Classes/Structs: `CamelCase`
  - Functions: `camelBack`
  - Variables: `lower_case`
  - Constants: `UPPER_CASE`
  - Macros: `UPPER_CASE`

**Format code**:

```bash
# Format all files
make format

# Format specific file
clang-format -i myfile.cu

# Check formatting without modifying
clang-format --dry-run --Werror *.cu
```

### Python Code

We use **black** for Python formatting.

```bash
black --line-length=100 *.py
```

### CMake Files

We use **cmake-format** for CMakeLists.txt formatting.

```bash
cmake-format -i CMakeLists.txt
```

## Testing Strategy

### Unit Tests

We use **Google Test** for unit testing. Tests are located in the `tests/` directory.

**Test Structure**:

```cpp
// Test fixture for shared setup
class MyTest : public ::testing::Test {
protected:
    void SetUp() override {
        // Setup code
    }

    void TearDown() override {
        // Cleanup code
    }
};

// Individual test cases
TEST_F(MyTest, TestName) {
    EXPECT_EQ(1 + 1, 2);
    ASSERT_TRUE(condition);
}
```

**Running Tests**:

```bash
# All tests
make test

# Verbose output
cd build/build/Release && ctest --verbose

# Specific test
./build/build/Release/test_vector_add --gtest_filter="VectorAddTest.SmallVector"

# List all tests
./build/build/Release/test_vector_add --gtest_list_tests
```

**Test Coverage**:

```bash
# Build with coverage
cmake --preset conan-release -DCMAKE_BUILD_TYPE=Debug -DCMAKE_CXX_FLAGS="--coverage"
cmake --build build/build/Debug

# Run tests
cd build/build/Debug && ctest

# Generate coverage report
gcov *.gcda
```

### Integration Tests

Integration tests verify the complete application flow. Add them to `tests/` with the prefix
`integration_`.

### Performance Tests

Performance tests measure execution time and throughput. Include them in unit tests but mark them
differently:

```cpp
TEST_F(VectorAddTest, DISABLED_PerformanceBenchmark) {
    // This test is disabled by default
    // Run with: --gtest_also_run_disabled_tests
}
```

## Static Analysis

### clang-tidy

**Run Analysis**:

```bash
# All files
make lint

# Specific file
clang-tidy -p build myfile.cu

# Fix issues automatically (use with caution)
clang-tidy -p build --fix myfile.cu
```

**Common Checks**:

- `modernize-*`: Use modern C++ features
- `readability-*`: Improve code readability
- `performance-*`: Performance improvements
- `bugprone-*`: Detect potential bugs
- `cppcoreguidelines-*`: C++ Core Guidelines

**Disable Specific Warning**:

```cpp
// NOLINTNEXTLINE(check-name)
problematic_code();
```

### Custom Static Analysis

Create custom checks by adding scripts to `scripts/`:

```bash
#!/bin/bash
# scripts/custom_check.sh
# Add your custom validation logic
```

## Memory Analysis

### CPU Memory with Valgrind

**Basic Check**:

```bash
valgrind --leak-check=full ./build/build/Release/vectoradd
```

**Advanced Options**:

```bash
# Track origins of uninitialized values
valgrind --track-origins=yes ./vectoradd

# Generate suppression file
valgrind --gen-suppressions=all ./vectoradd 2>&1 | tee suppressions.txt

# Use suppression file
valgrind --suppressions=suppressions.txt ./vectoradd
```

**Common Issues**:

- **Definitely lost**: Real memory leak - must fix
- **Indirectly lost**: Child of leaked memory
- **Possibly lost**: Pointer manipulation detected
- **Still reachable**: Memory not freed at exit (often acceptable)

### GPU Memory with cuda-memcheck

**Available Tools**:

```bash
# Memory errors
cuda-memcheck --tool memcheck ./vectoradd

# Race conditions
cuda-memcheck --tool racecheck ./vectoradd

# Shared memory race detection
cuda-memcheck --tool racecheck --racecheck-report all ./vectoradd

# Check API usage
cuda-memcheck --tool initcheck ./vectoradd

# Synchronization errors
cuda-memcheck --tool synccheck ./vectoradd
```

**Common GPU Memory Issues**:

- Out-of-bounds access
- Uninitialized memory reads
- Race conditions in shared memory
- Missing synchronization

### Modern CUDA Debugging with compute-sanitizer

CUDA 11.4+ provides `compute-sanitizer`, a more advanced debugging tool:

```bash
# Memory check
compute-sanitizer --tool memcheck ./vectoradd

# Report all races (slower but comprehensive)
compute-sanitizer --tool racecheck --racecheck-report all ./vectoradd

# Save report to file
compute-sanitizer --tool memcheck --log-file report.txt ./vectoradd
```

## Documentation

### Code Documentation with Doxygen

**Documentation Style**:

```cpp
/**
 * @brief Brief description of the function
 *
 * Detailed description with more information about what the function does,
 * its algorithm, and any important notes.
 *
 * @param input Description of input parameter
 * @param size Number of elements
 * @return Description of return value
 *
 * @note Important note about usage
 * @warning Warning about potential issues
 * @see RelatedFunction
 *
 * @code
 * // Example usage
 * myFunction(data, 100);
 * @endcode
 */
void myFunction(float* input, int size);
```

**Generate Documentation**:

```bash
# Generate HTML documentation
make docs

# Open documentation
xdg-open docs/html/index.html
```

**Documentation Best Practices**:

- Document all public APIs
- Include usage examples
- Explain complex algorithms
- Document CUDA kernel launch parameters
- Note thread safety requirements
- Specify memory ownership

### README and Guides

- **README.md**: Quick start and overview
- **DEVELOPMENT.md**: This file - detailed development guide
- **CONTRIBUTING.md**: Contribution guidelines
- **API.md**: API reference (if needed)

## Debugging

### CPU Debugging with GDB

```bash
# Build in debug mode
make debug

# Start GDB
gdb ./build/build/Debug/vectoradd

# Common GDB commands
(gdb) break main          # Set breakpoint
(gdb) run                 # Run program
(gdb) next                # Step over
(gdb) step                # Step into
(gdb) continue            # Continue execution
(gdb) print variable      # Print variable value
(gdb) backtrace          # Show call stack
```

### GPU Debugging with cuda-gdb

```bash
# Build with debug symbols
cmake --preset conan-debug -DCMAKE_BUILD_TYPE=Debug
cmake --build build/build/Debug

# Start cuda-gdb
cuda-gdb ./build/build/Debug/vectoradd

# CUDA-specific commands
(cuda-gdb) info cuda kernels    # List running kernels
(cuda-gdb) cuda thread          # Switch to CUDA thread
(cuda-gdb) cuda block           # Show current block
(cuda-gdb) cuda kernel          # Show kernel info
```

### Nsight Tools

**Nsight Systems** (system-wide profiling):

```bash
nsys profile --stats=true ./vectoradd
nsys-ui  # Open GUI
```

**Nsight Compute** (kernel profiling):

```bash
ncu --set full ./vectoradd
ncu-ui  # Open GUI
```

## Performance Optimization

### Profiling

1. **Profile First**: Always profile before optimizing
1. **Identify Hotspots**: Focus on code that takes the most time
1. **Measure Impact**: Verify improvements with benchmarks

### CUDA Optimization Checklist

- [ ] Maximize occupancy (use `--ptxas-options=-v`)
- [ ] Coalesce global memory accesses
- [ ] Use shared memory for frequently accessed data
- [ ] Minimize divergence (branch uniformity)
- [ ] Use appropriate memory spaces (shared, constant, texture)
- [ ] Optimize block and grid dimensions
- [ ] Use CUDA streams for concurrency
- [ ] Profile with Nsight Compute
- [ ] Consider using cuBLAS/cuFFT for standard operations

### Compiler Optimizations

```cmake
# Release build flags
set(CMAKE_CXX_FLAGS_RELEASE "-O3 -march=native -DNDEBUG")
set(CMAKE_CUDA_FLAGS_RELEASE "-O3 -use_fast_math")

# Debug with optimization
set(CMAKE_CXX_FLAGS_RELWITHDEBINFO "-O2 -g")
```

## CI/CD Integration

### GitHub Actions Example

```yaml
name: CI

on: [push, pull_request]

jobs:
  build-and-test:
    runs-on: ubuntu-latest

    steps:
    - uses: actions/checkout@v3

    - name: Install dependencies
      run: |
        sudo apt update
        sudo apt install -y cmake ninja-build

    - name: Build
      run: make build

    - name: Run tests
      run: make test

    - name: Check formatting
      run: |
        make format
        git diff --exit-code

    - name: Run linter
      run: make lint

    - name: Generate documentation
      run: make docs

    - name: Upload documentation
      uses: actions/upload-artifact@v3
      with:
        name: documentation
        path: docs/html/
```

### Pre-commit in CI

```yaml
- name: Pre-commit checks
  run: |
    pip install pre-commit
    pre-commit run --all-files
```

## Best Practices Summary

1. **Always run tests** after making changes
1. **Format code** before committing
1. **Run static analysis** regularly
1. **Check for memory leaks** in production code
1. **Document public APIs** with Doxygen comments
1. **Use pre-commit hooks** to catch issues early
1. **Profile before optimizing** CUDA kernels
1. **Keep tests fast** (\< 1 second for unit tests)
1. **Use meaningful commit messages**
1. **Review your own code** before submitting

## Getting Help

- Check existing documentation first
- Search closed issues on GitHub
- Ask in project discussions
- Create a detailed issue with reproducible example

## Additional Resources

- [CUDA C++ Programming Guide](https://docs.nvidia.com/cuda/cuda-c-programming-guide/)
- [CUDA Best Practices Guide](https://docs.nvidia.com/cuda/cuda-c-best-practices-guide/)
- [Google C++ Style Guide](https://google.github.io/styleguide/cppguide.html)
- [Google Test Documentation](https://google.github.io/googletest/)
- [CMake Documentation](https://cmake.org/documentation/)
- [Doxygen Manual](https://www.doxygen.nl/manual/)
