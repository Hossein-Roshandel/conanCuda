# Contributing to ConanCuda

Thank you for your interest in contributing to ConanCuda! This document provides guidelines and information for contributors.

## Table of Contents

- [Code of Conduct](#code-of-conduct)
- [Getting Started](#getting-started)
- [Development Environment](#development-environment)
- [How to Contribute](#how-to-contribute)
- [Pull Request Process](#pull-request-process)
- [Coding Standards](#coding-standards)
- [Testing Guidelines](#testing-guidelines)
- [Documentation](#documentation)

## Code of Conduct

This project adheres to a code of conduct that we expect all contributors to follow:

- **Be respectful**: Treat everyone with respect and kindness
- **Be collaborative**: Work together constructively
- **Be inclusive**: Welcome newcomers and help them get started
- **Be patient**: Understand that everyone has different experience levels

## Getting Started

### Prerequisites

Before contributing, ensure you have:

- **Git**: For version control
- **Docker**: For dev container development (recommended)
- **VS Code**: With Dev Containers extension (recommended)
- **NVIDIA GPU**: For testing CUDA functionality
- **Basic knowledge**: C++, CUDA, CMake, and Conan

### Fork and Clone

1. **Fork the repository** on GitHub
2. **Clone your fork**:
   ```bash
   git clone https://github.com/YOUR_USERNAME/conanCuda.git
   cd conanCuda
   ```
3. **Add upstream remote**:
   ```bash
   git remote add upstream https://github.com/Hossein-Roshandel/conanCuda.git
   ```

## Development Environment

### Option 1: Dev Container (Recommended)

1. **Open in VS Code**:
   ```bash
   code .
   ```

2. **Reopen in Container**: VS Code will prompt, or use `Ctrl+Shift+P` → "Dev Containers: Reopen in Container"

3. **Verify setup**:
   ```bash
   make build
   ./build/build/Release/vectoradd
   ```

### Option 2: Local Development

1. **Install dependencies**:
   ```bash
   sudo apt update
   sudo apt install cmake nvidia-cuda-toolkit python3-pip build-essential -y
   curl -LsSf https://astral.sh/uv/install.sh | sh
   uv tool install conan
   ```

2. **Build and test**:
   ```bash
   make build
   ./build/build/Release/vectoradd
   ```

## How to Contribute

### Types of Contributions

We welcome various types of contributions:

1. **🐛 Bug Reports**: Report issues you encounter
2. **✨ Feature Requests**: Suggest new features or improvements
3. **📝 Documentation**: Improve documentation and examples
4. **🧪 Examples**: Add new CUDA examples or use cases
5. **🔧 Build System**: Improve CMake, Conan, or container setup
6. **🚀 Performance**: Optimize existing code
7. **🛠️ Tooling**: Improve development tools and workflows

### Reporting Issues

Before creating an issue:

1. **Search existing issues** to avoid duplicates
2. **Use issue templates** when available
3. **Provide system information**:
   - OS and version
   - CUDA toolkit version
   - GPU model and driver version
   - CMake and Conan versions

**Good issue example**:
```
**Problem**: Vector addition example crashes with "illegal memory access"

**Environment**:
- Ubuntu 22.04
- CUDA 12.0
- RTX 3080
- Driver 525.60.11

**Steps to reproduce**:
1. Clone repository
2. Run `make build`
3. Execute `./build/build/Release/vectoradd`

**Expected**: Program completes successfully
**Actual**: Segmentation fault at line 45

**Additional info**: Works fine with smaller array sizes (< 1000 elements)
```

### Suggesting Features

When proposing features:

1. **Explain the use case**: Why is this needed?
2. **Describe the solution**: What should it do?
3. **Consider alternatives**: Are there other approaches?
4. **Think about impact**: How does it affect existing functionality?

## Pull Request Process

### Before You Start

1. **Check existing work**: Look for related PRs or issues
2. **Discuss major changes**: Open an issue first for significant modifications
3. **Create a branch**: Use descriptive branch names
   ```bash
   git checkout -b feature/cuda-memory-pool
   git checkout -b fix/cmake-cuda-flags
   git checkout -b docs/installation-guide
   ```

### Development Workflow

1. **Keep your fork updated**:
   ```bash
   git fetch upstream
   git checkout main
   git merge upstream/main
   ```

2. **Create feature branch**:
   ```bash
   git checkout -b your-feature-branch
   ```

3. **Make changes and commit**:
   ```bash
   git add .
   git commit -m "Add CUDA memory pool example"
   ```

4. **Test your changes**:
   ```bash
   make build
   # Test all executables
   ./build/build/Release/conancuda
   ./build/build/Release/vectoradd
   ```

5. **Push and create PR**:
   ```bash
   git push origin your-feature-branch
   ```

### Pull Request Guidelines

**PR Title Format**:
- `feat: add CUDA memory pool example`
- `fix: resolve CMake CUDA flags issue`
- `docs: update installation instructions`
- `refactor: improve error handling`

**PR Description Template**:
```markdown
## What
Brief description of what this PR does.

## Why
Explanation of why this change is needed.

## How
Technical details of the implementation.

## Testing
- [ ] Builds successfully
- [ ] Examples run without errors
- [ ] New tests pass (if applicable)
- [ ] Documentation updated (if applicable)

## Screenshots/Output
If applicable, add screenshots or output examples.
```

## Coding Standards

### C++ Style

Follow modern C++ best practices:

```cpp
// Use meaningful names
int numElements = 50000;  // Good
int n = 50000;           // Avoid

// Use const and auto appropriately
const auto deviceCount = cuda::device::count();

// Prefer RAII and smart pointers
auto deviceMemory = cuda::memory::make_unique_span<float>(device, size);

// Use consistent indentation (4 spaces)
if (condition) {
    doSomething();
    doSomethingElse();
}
```

### CUDA Guidelines

- **Error checking**: Always check CUDA errors
- **Memory management**: Use RAII wrappers when possible
- **Kernel optimization**: Document performance considerations
- **Thread safety**: Be clear about thread safety guarantees

```cuda
// Good: Check for errors
__global__ void myKernel(float* data, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {  // Bounds checking
        data[idx] *= 2.0f;
    }
}

// Good: Document kernel requirements
/**
 * Vector addition kernel
 * @param A Input vector A
 * @param B Input vector B  
 * @param C Output vector C (A + B)
 * @param numElements Number of elements to process
 * 
 * Requirements:
 * - All pointers must be valid device memory
 * - numElements > 0
 * - Launch with (numElements + blockSize - 1) / blockSize blocks
 */
__global__ void vectorAdd(const float *A, const float *B, float *C, int numElements);
```

### CMake Best Practices

- Use modern CMake (3.5+)
- Explicit target-based linking
- Clear variable naming
- Comments for complex logic

```cmake
# Good: Modern CMake
add_executable(vectoradd vectorAdd.cu)
target_link_libraries(vectoradd 
    PRIVATE 
        cuda-api-wrappers::runtime-and-driver 
        cuda
)

# Good: Clear configuration
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)
```

## Testing Guidelines

### Manual Testing

Before submitting:

1. **Build test**:
   ```bash
   make build
   ```

2. **Run examples**:
   ```bash
   ./build/build/Release/conancuda
   ./build/build/Release/vectoradd
   ```

3. **Test different configurations**:
   - Debug and Release builds
   - Different CUDA architectures
   - Various input sizes

### Adding Tests

When adding new functionality:

1. **Create test cases** for new features
2. **Test edge cases** (empty inputs, large sizes, etc.)
3. **Performance tests** for CUDA kernels
4. **Cross-platform testing** when possible

## Documentation

### Code Documentation

- **Header comments**: Explain purpose and usage
- **Inline comments**: Clarify complex algorithms
- **Parameter documentation**: Document inputs/outputs
- **Examples**: Provide usage examples

### README Updates

When adding features:

1. **Update feature list**
2. **Add usage examples**
3. **Update installation instructions** if needed
4. **Add troubleshooting notes**

### Example Documentation

```cpp
/**
 * Performs parallel vector addition on GPU
 * 
 * @param h_A Host vector A (input)
 * @param h_B Host vector B (input)  
 * @param h_C Host vector C (output, A + B)
 * @param numElements Number of elements in vectors
 * 
 * @return true if successful, false on error
 * 
 * Example:
 * ```cpp
 * std::vector<float> a(1000, 1.0f);
 * std::vector<float> b(1000, 2.0f);
 * std::vector<float> c(1000);
 * 
 * if (vectorAdd(a, b, c, 1000)) {
 *     std::cout << "Success: c[0] = " << c[0] << std::endl;  // Should be 3.0
 * }
 * ```
 */
bool vectorAdd(const std::vector<float>& h_A, 
               const std::vector<float>& h_B, 
               std::vector<float>& h_C, 
               int numElements);
```

## Review Process

### What We Look For

- **Functionality**: Does it work as intended?
- **Code Quality**: Is it readable and maintainable?
- **Performance**: Are there any performance implications?
- **Documentation**: Is it properly documented?
- **Testing**: Has it been adequately tested?
- **Compatibility**: Does it work across different systems?

### Review Timeline

- **Initial review**: Within 48 hours
- **Follow-up**: Based on complexity
- **Merge**: After approval and CI success

## Getting Help

If you need help:

1. **Check documentation**: README, issues, and code comments
2. **Search existing issues**: Your question might be answered
3. **Ask in discussions**: Use GitHub Discussions for questions
4. **Contact maintainers**: For urgent issues

## Recognition

Contributors will be:

- **Listed in acknowledgments**
- **Tagged in release notes** for significant contributions
- **Invited as collaborators** for consistent contributors

## Questions?

Feel free to reach out:

- **GitHub Issues**: For bugs and feature requests
- **GitHub Discussions**: For questions and general discussion
- **Email**: For private matters

Thank you for contributing to ConanCuda! 🚀