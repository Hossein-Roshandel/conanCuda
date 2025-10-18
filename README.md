# CUDA C++ Development with Conan and CMake

A professional-grade boilerplate repository for CUDA C++ development using modern tools and best
practices. This setup provides a complete, reproducible development environment for GPU computing
projects with integrated testing, documentation, and code quality tools.

## Features

- 🚀 **CUDA C++ Development**: Ready-to-use CUDA development environment with cuda-api-wrappers
- 📦 **Conan Package Management**: Dependency management with Conan 2.x
- 🏗️ **CMake Build System**: Modern CMake configuration with presets
- 🐳 **Dev Container Support**: Containerized development environment
- 🧪 **Unit Testing**: Google Test integration for comprehensive testing
- 📚 **Documentation**: Doxygen for API documentation generation
- 🔍 **Code Quality**: clang-format, clang-tidy, and pre-commit hooks
- 🛡️ **Memory Safety**: Valgrind, cuda-memcheck, and compute-sanitizer integration
- ⚡ **Quick Setup**: Get started with a few commands

## Prerequisites

### Local Development

- **CUDA Toolkit**: NVIDIA CUDA Toolkit 11.0+
- **CMake**: Version 3.5.0 or higher
- **Python**: 3.13+ (for Conan)
- **C++ Compiler**: GCC 13.3+ or equivalent with C++17 support
- **GPU**: NVIDIA GPU with compute capability 3.5+

### Container Development

- **Docker**: For dev container support
- **VS Code**: With Dev Containers extension
- **NVIDIA Container Toolkit** (optional): For GPU access in containers -
  [Setup Guide](.devcontainer/GPU_SETUP.md)

> **Note**: The dev container works without GPU access. You can write and build CUDA code, but to
> run CUDA programs you'll need to enable GPU passthrough. See
> [GPU Setup Guide](.devcontainer/GPU_SETUP.md) for instructions.

## Quick Start

### Option 1: Using Dev Container (Recommended)

1. **Clone the repository**:

   ```bash
   git clone https://github.com/Hossein-Roshandel/conanCuda.git
   cd conanCuda
   ```

1. **Open in VS Code**:

   ```bash
   code .
   ```

1. **Reopen in Container**: VS Code will prompt to reopen in dev container, or use `Ctrl+Shift+P` →
   "Dev Containers: Reopen in Container"

1. **Build and run**:

   ```bash
   make build
   ./build/build/Release/vectoradd
   ```

### Option 2: Local Development

1. **Install dependencies**:

   ```bash
   # Ubuntu/Debian
   sudo apt update
   sudo apt install cmake nvidia-cuda-toolkit python3-pip -y

   # Install uv (Python package manager)
   curl -LsSf https://astral.sh/uv/install.sh | sh

   # Install Conan
   uv tool install conan
   ```

1. **Verify installations**:

   ```bash
   cmake --version      # Should show 3.5.0+
   nvcc --version       # Should show CUDA toolkit version
   conan --version      # Should show Conan 2.x
   ```

1. **Clone and build**:

   ```bash
   git clone https://github.com/Hossein-Roshandel/conanCuda.git
   cd conanCuda
   make build
   ```

1. **Run examples**:

   ```bash
   # Run simple C++ example
   ./build/build/Release/conancuda

   # Run CUDA vector addition example
   ./build/build/Release/vectoradd
   ```

## Project Structure

```
conanCuda/
├── .devcontainer/           # Dev container configuration
│   ├── devcontainer.json
│   ├── docker-compose.yml
│   ├── post_create.sh      # Container initialization script
│   └── post_start.sh
├── build/                   # Build artifacts (generated)
├── docs/                    # Generated documentation
│   └── html/               # Doxygen HTML output
├── scripts/                 # Utility scripts
│   ├── memory_check.sh     # Memory analysis script
│   └── static_analysis.sh  # Static code analysis script
├── tests/                   # Unit tests
│   └── test_vector_add.cu  # Vector addition tests
├── .clang-format           # Code formatting rules
├── .clang-tidy            # Static analysis rules
├── .pre-commit-config.yaml # Pre-commit hooks configuration
├── CMakeLists.txt          # CMake configuration
├── CMakePresets.json       # CMake presets
├── CMakeUserPresets.json   # User-specific CMake presets
├── conanfile.py           # Conan dependencies
├── Doxyfile               # Doxygen configuration
├── main.cpp               # Simple C++ example
├── vectorAdd.cu           # CUDA vector addition example
├── vector_operations.cu   # Shared CUDA kernel implementation
├── vector_operations.cuh  # CUDA kernel header
├── Makefile               # Build shortcuts
├── pyproject.toml         # Python/uv configuration
├── README.md              # This file
├── CONTRIBUTING.md        # Contribution guidelines
└── LICENSE                # MIT license
```

## Building the Project

### Using Make (Recommended)

```bash
# Build the project
make build

# Run tests
make test

# Run specific example
make run

# Format code
make format

# Run static analysis
make lint

# Check for memory issues
make memory-check

# Generate documentation
make docs

# Clean build artifacts
make clean

# Show all available targets
make help
```

### Manual Build Process

```bash
# Install dependencies with Conan
uv run conan install . --output-folder=build --build=missing

# Configure CMake with tests enabled
cmake --preset conan-release -DBUILD_TESTS=ON

# Build
cmake --build build/build/Release

# Run tests
cd build/build/Release && ctest --verbose
```

### Available Targets

- `conancuda`: Simple "Hello World" C++ application
- `vectoradd`: CUDA vector addition example using cuda-api-wrappers
- `test_vector_add`: Unit tests for vector addition kernel
- `vector_operations`: Shared library with CUDA kernels

## Adding Dependencies

### Adding Conan Packages

1. **Edit `conanfile.py`**:

   ```python
   def requirements(self):
       self.requires("cuda-api-wrappers/0.8.0")
       self.requires("your-package/version")  # Add your package here
   ```

1. **Rebuild**:

   ```bash
   make build
   ```

1. **Use in CMakeLists.txt**:

   ```cmake
   find_package(your-package REQUIRED)
   target_link_libraries(your_target your-package::your-package)
   ```

### Popular CUDA/C++ Packages

- `cuda-api-wrappers`: Modern C++ CUDA API wrappers
- `gtest`: Google Test framework (build dependency)
- `thrust`: CUDA parallel algorithms library
- `cub`: CUDA primitives library
- `eigen`: Linear algebra library
- `opencv`: Computer vision library
- `boost`: C++ utilities

## Development Workflow

### Recommended Development Cycle

1. **Write Code**: Edit your `.cpp`, `.cu`, or `.cuh` files
1. **Format**: `make format` to ensure consistent style
1. **Build**: `make build` to compile the project
1. **Test**: `make test` to run unit tests
1. **Lint**: `make lint` to catch potential issues
1. **Memory Check**: `make memory-check` for production code
1. **Document**: Add Doxygen comments for public APIs
1. **Commit**: Pre-commit hooks run automatically

### Quick Development Commands

```bash
# Full quality check before commit
make format && make build && make test && make lint

# Build and run
make build && make run

# Debug mode build
make debug

# Check system information
make info
```

### CMake Options

Build with additional features:

```bash
# Enable tests (default: ON)
cmake --preset conan-release -DBUILD_TESTS=ON

# Enable documentation generation
cmake --preset conan-release -DBUILD_DOCS=ON

# Enable clang-tidy during build
cmake --preset conan-release -DENABLE_CLANG_TIDY=ON

# Combine options
cmake --preset conan-release -DBUILD_TESTS=ON -DBUILD_DOCS=ON
```

## Examples

### Vector Addition (vectorAdd.cu)

A complete CUDA example demonstrating:

- Modern C++ CUDA API wrappers
- Automatic memory management with RAII
- Simplified kernel launch configuration
- Device-host memory transfers
- Error checking and verification

Run with:

```bash
./build/build/Release/vectoradd
```

Expected output:

```
[Vector addition of 50000 elements]
CUDA kernel launch with 196 blocks of 256 threads each
Test PASSED
SUCCESS
```

## Code Quality Tools

### Code Formatting with clang-format

Automatically format your code to maintain consistent style:

```bash
# Format all source files
make format

# Or manually
clang-format -i *.cpp *.cu *.cuh *.hpp
```

Configuration is in `.clang-format` (based on Google style with CUDA adaptations).

### Static Analysis with clang-tidy

Detect potential bugs, performance issues, and style violations:

```bash
# Run static analysis
make lint

# Or manually
./scripts/static_analysis.sh
```

Configuration is in `.clang-tidy` with comprehensive checks for modern C++ and CUDA.

### Pre-commit Hooks

Automatically run checks before each commit:

```bash
# Install pre-commit hooks
make pre-commit

# Or manually
pip install pre-commit
pre-commit install

# Run on all files
pre-commit run --all-files
```

The `.pre-commit-config.yaml` includes:

- Code formatting (clang-format, black)
- Linting (clang-tidy, flake8, shellcheck)
- File checks (trailing whitespace, merge conflicts)
- Security (secret detection)
- Documentation (markdown formatting)

### Memory Analysis

#### Valgrind (CPU Memory)

Detect memory leaks and invalid memory access in CPU code:

```bash
# Run comprehensive memory check
make memory-check

# Or manually
valgrind --leak-check=full --show-leak-kinds=all ./build/build/Release/vectoradd
```

#### cuda-memcheck (GPU Memory)

Check for CUDA-specific memory errors:

```bash
# Included in make memory-check
# Or run manually
cuda-memcheck --leak-check full --tool memcheck ./build/build/Release/vectoradd
```

#### compute-sanitizer (Modern CUDA Tool)

CUDA 11.4+ tool for comprehensive GPU debugging:

```bash
# Memory check
compute-sanitizer --tool memcheck --leak-check full ./build/build/Release/vectoradd

# Race condition detection
compute-sanitizer --tool racecheck ./build/build/Release/vectoradd

# Initialization check
compute-sanitizer --tool initcheck ./build/build/Release/vectoradd

# Synchronization check
compute-sanitizer --tool synccheck ./build/build/Release/vectoradd
```

All checks are automated in `scripts/memory_check.sh`.

### Unit Testing with Google Test

Write and run comprehensive unit tests:

```bash
# Run all tests
make test

# Run tests with verbose output
cd build/build/Release && ctest --verbose

# Run specific test
./build/build/Release/test_vector_add --gtest_filter="VectorAddTest.SmallVector"
```

Test files are in the `tests/` directory. Example test structure:

```cpp
TEST_F(VectorAddTest, SmallVector) {
    // Test small vector addition
}

TEST_F(VectorAddTest, LargeVector) {
    // Test large vector addition
}
```

### Documentation Generation with Doxygen

Generate API documentation from code comments:

```bash
# Generate documentation
make docs

# View documentation
xdg-open docs/html/index.html  # Linux
open docs/html/index.html       # macOS
start docs/html/index.html      # Windows
```

Documentation includes:

- Class and function documentation
- Call graphs and dependency diagrams
- Source code browser
- CUDA kernel documentation

Configuration is in `Doxyfile` with CUDA-specific settings.

## Development Environment

### VS Code Extensions (Auto-installed in dev container)

- **C/C++**: IntelliSense and debugging
- **CMake Tools**: CMake integration
- **Python**: Python language support
- **Docker**: Container management
- **YAML**: Configuration file support

### Dev Container Features

- **CUDA Toolkit**: Pre-installed and configured
- **CMake & Conan**: Latest versions
- **Development Tools**: GDB, Git, etc.
- **GPU Access**: NVIDIA runtime support

## GPU Requirements

### Minimum Requirements

- **NVIDIA GPU**: Compute Capability 3.5+
- **CUDA Drivers**: Compatible with your CUDA toolkit version
- **Memory**: 2GB+ VRAM recommended

### Testing GPU Access

```bash
# Check GPU availability
nvidia-smi

# Test CUDA installation
nvcc --version

# Run the vector addition example
./build/build/Release/vectoradd
```

## Troubleshooting

### Common Issues

1. **"No CUDA devices on this system"**

   - Verify GPU drivers: `nvidia-smi`
   - Check CUDA installation: `nvcc --version`
   - Ensure GPU compute capability ≥ 3.5

1. **Conan package not found**

   - Update Conan: `uv tool install --upgrade conan`
   - Check package availability: `conan search package-name`
   - Try different version: Update version in `conanfile.py`

1. **CMake configuration failed**

   - Clean build: `rm -rf build/`
   - Reinstall dependencies: `make build`
   - Check CMake version: `cmake --version`

1. **Dev container won't start**

   - Ensure Docker is running
   - Check NVIDIA Container Toolkit installation
   - Verify devcontainer.json syntax

### Getting Help

- Check [Issues](https://github.com/Hossein-Roshandel/conanCuda/issues) for known problems
- Create a new issue with your error message and system info
- See [CONTRIBUTING.md](CONTRIBUTING.md) for development guidelines

## Performance Tips

- **Memory Management**: Use cuda-api-wrappers for automatic cleanup
- **Kernel Optimization**: Profile with `nvprof`, `nsys`, or Nsight Compute
- **Block Size**: Experiment with different block sizes (64, 128, 256, 512)
- **Memory Patterns**: Ensure coalesced memory access
- **Shared Memory**: Use shared memory for frequently accessed data
- **Occupancy**: Check occupancy with `--ptxas-options=-v`
- **Stream Parallelism**: Use CUDA streams for concurrent operations

## Continuous Integration

### Pre-commit Hooks

Ensure code quality before every commit:

- ✅ Code formatting
- ✅ Trailing whitespace removal
- ✅ YAML/JSON validation
- ✅ Static analysis
- ✅ Secret detection

### Recommended CI Pipeline

```yaml
# Example GitHub Actions workflow
- name: Build
  run: make build

- name: Test
  run: make test

- name: Lint
  run: make lint

- name: Format Check
  run: |
    make format
    git diff --exit-code

- name: Documentation
  run: make docs
```

## Tools Reference

### Make Targets Summary

| Command             | Description                         |
| ------------------- | ----------------------------------- |
| `make build`        | Build the project with dependencies |
| `make test`         | Run all unit tests                  |
| `make run`          | Run vector addition example         |
| `make clean`        | Remove build artifacts              |
| `make format`       | Format code with clang-format       |
| `make lint`         | Run static analysis                 |
| `make memory-check` | Check for memory leaks              |
| `make docs`         | Generate documentation              |
| `make pre-commit`   | Install pre-commit hooks            |
| `make debug`        | Build in debug mode                 |
| `make info`         | Show build environment info         |
| `make help`         | Show all available targets          |

### Code Quality Checklist

Before submitting code:

- [ ] Code compiles without warnings
- [ ] All tests pass
- [ ] Code is formatted (`make format`)
- [ ] No linting errors (`make lint`)
- [ ] No memory leaks (`make memory-check`)
- [ ] Documentation is updated
- [ ] Pre-commit hooks installed and passing
- [ ] New features have unit tests

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Contributing

Please read [CONTRIBUTING.md](CONTRIBUTING.md) for guidelines on contributing to this project.

## Acknowledgments

- NVIDIA CUDA samples and documentation
- cuda-api-wrappers library by Eyal Rozenberg
- Conan package manager community
- CMake development team
