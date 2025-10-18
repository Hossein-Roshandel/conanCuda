# Quick Reference Card

## 🚀 Essential Commands

### Build and Run

```bash
make build           # Build project with dependencies
make run             # Run vector addition example
make test            # Run all tests
make clean           # Clean build artifacts
```

### Code Quality

```bash
make format          # Format code with clang-format
make lint            # Run static analysis with clang-tidy
make memory-check    # Run memory analysis (Valgrind + cuda-memcheck)
```

### Documentation

```bash
make docs            # Generate Doxygen documentation
make info            # Show build environment information
make help            # Show all available targets
```

### Development Setup

```bash
make pre-commit      # Install pre-commit hooks
make debug           # Build in debug mode
```

## 📁 Project Structure

```
conanCuda/
├── vectorAdd.cu              # Main CUDA example (documented)
├── vector_operations.cu/h    # Shared kernel code
├── tests/                    # Unit tests
├── scripts/                  # Utility scripts
├── docs/                     # Generated documentation
├── .clang-format            # Code style rules
├── .clang-tidy              # Static analysis rules
├── .pre-commit-config.yaml  # Pre-commit hooks
├── Doxyfile                 # Documentation config
├── CMakeLists.txt           # Build configuration
├── Makefile                 # Quick commands
└── README.md                # Full documentation
```

## 🔧 CMake Build Options

```bash
# Enable tests (default: ON)
cmake --preset conan-release -DBUILD_TESTS=ON

# Enable documentation
cmake --preset conan-release -DBUILD_DOCS=ON

# Enable clang-tidy during build
cmake --preset conan-release -DENABLE_CLANG_TIDY=ON
```

## 🧪 Testing

```bash
# Run all tests
make test

# Run with verbose output
cd build/build/Release && ctest --verbose

# Run specific test
./build/build/Release/test_vector_add --gtest_filter="VectorAddTest.*"

# List available tests
./build/build/Release/test_vector_add --gtest_list_tests
```

## 🔍 Code Analysis

### Static Analysis

```bash
# Analyze all files
make lint

# Analyze specific file
clang-tidy -p build myfile.cu

# Auto-fix issues (careful!)
clang-tidy -p build --fix myfile.cu
```

### Memory Analysis

```bash
# All checks (Valgrind + CUDA)
make memory-check

# CPU memory only
valgrind --leak-check=full ./build/build/Release/vectoradd

# GPU memory only
cuda-memcheck ./build/build/Release/vectoradd

# Modern CUDA debugging
compute-sanitizer --tool memcheck ./build/build/Release/vectoradd
```

## 📚 Documentation

### View Documentation

```bash
# Generate docs
make docs

# Open in browser (Linux)
xdg-open docs/html/index.html

# Open in browser (macOS)
open docs/html/index.html
```

### Doxygen Comment Style

```cpp
/**
 * @brief Brief function description
 *
 * @param input Description of parameter
 * @return Description of return value
 *
 * @note Usage notes
 * @warning Important warnings
 */
void myFunction(int input);
```

## 🎨 Code Formatting

```bash
# Format all files
make format

# Format specific file
clang-format -i myfile.cu

# Check without modifying
clang-format --dry-run myfile.cu
```

## 🪝 Git Hooks

```bash
# Install hooks
make pre-commit

# Run hooks manually
pre-commit run --all-files

# Run specific hook
pre-commit run clang-format --all-files

# Skip hooks for emergency commit
git commit --no-verify
```

## 🐛 Debugging

### CPU Debugging

```bash
# Build in debug mode
make debug

# Start GDB
gdb ./build/build/Debug/vectoradd

# Common GDB commands
break main      # Set breakpoint
run            # Run program
next           # Step over
step           # Step into
continue       # Continue execution
print var      # Print variable
backtrace      # Show stack
```

### GPU Debugging

```bash
# Start cuda-gdb
cuda-gdb ./build/build/Debug/vectoradd

# CUDA-specific commands
info cuda kernels    # List kernels
cuda thread         # Switch to CUDA thread
cuda block          # Current block info
```

## 📊 Performance Profiling

```bash
# System-wide profiling
nsys profile --stats=true ./vectoradd

# Kernel profiling
ncu --set full ./vectoradd

# Legacy profiler
nvprof ./vectoradd
```

## 🔗 Quick Links

- **Full Documentation**: [README.md](README.md)
- **Development Guide**: [DEVELOPMENT.md](DEVELOPMENT.md)
- **Feature Summary**: [FEATURES.md](FEATURES.md)
- **Contributing**: [CONTRIBUTING.md](CONTRIBUTING.md)

## 💡 Tips

1. **Always format before commit**: `make format`
1. **Run tests frequently**: `make test`
1. **Check for memory leaks**: `make memory-check` (before release)
1. **Keep docs updated**: Add Doxygen comments
1. **Use pre-commit hooks**: `make pre-commit`
1. **Profile before optimizing**: Use nsys/ncu
1. **Build in debug for debugging**: `make debug`
1. **Read DEVELOPMENT.md**: For advanced usage

## 🆘 Troubleshooting

```bash
# Clean everything and rebuild
make clean && make build

# Check environment
make info

# Update dependencies
uv run conan install . --output-folder=build --build=missing --update

# Verify CUDA installation
nvcc --version
nvidia-smi
```

## 📞 Getting Help

1. Check documentation (README.md, DEVELOPMENT.md)
1. Run `make help` for available commands
1. Check existing GitHub issues
1. Create new issue with details

______________________________________________________________________

**Remember**: Run `make help` to see all available commands!
