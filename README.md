# CUDA C++ Development with Conan and CMake

A boilerplate repository for CUDA C++ development using Conan package manager and CMake build system. This setup provides a modern, reproducible development environment for GPU computing projects.

## Features

- 🚀 **CUDA C++ Development**: Ready-to-use CUDA development environment
- 📦 **Conan Package Management**: Dependency management with Conan 2.x
- 🏗️ **CMake Build System**: Modern CMake configuration with presets
- 🐳 **Dev Container Support**: Containerized development environment
- 🔧 **GPU Programming Examples**: Vector addition example with cuda-api-wrappers
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
- **NVIDIA Container Toolkit** (optional): For GPU access in containers - [Setup Guide](.devcontainer/GPU_SETUP.md)

> **Note**: The dev container works without GPU access. You can write and build CUDA code, but to run CUDA programs you'll need to enable GPU passthrough. See [GPU Setup Guide](.devcontainer/GPU_SETUP.md) for instructions.

## Quick Start

### Option 1: Using Dev Container (Recommended)

1. **Clone the repository**:
   ```bash
   git clone https://github.com/Hossein-Roshandel/conanCuda.git
   cd conanCuda
   ```

2. **Open in VS Code**:
   ```bash
   code .
   ```

3. **Reopen in Container**: VS Code will prompt to reopen in dev container, or use `Ctrl+Shift+P` → "Dev Containers: Reopen in Container"

4. **Build and run**:
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

2. **Verify installations**:
   ```bash
   cmake --version      # Should show 3.5.0+
   nvcc --version       # Should show CUDA toolkit version
   conan --version      # Should show Conan 2.x
   ```

3. **Clone and build**:
   ```bash
   git clone https://github.com/Hossein-Roshandel/conanCuda.git
   cd conanCuda
   make build
   ```

4. **Run examples**:
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
│   └── docker-compose.yml
├── build/                   # Build artifacts (generated)
├── CMakeLists.txt          # CMake configuration
├── CMakePresets.json       # CMake presets
├── CMakeUserPresets.json   # User-specific CMake presets
├── conanfile.py           # Conan dependencies
├── main.cpp               # Simple C++ example
├── vectorAdd.cu           # CUDA vector addition example
├── Makefile               # Build shortcuts
├── pyproject.toml         # Python/uv configuration
├── README.md              # This file
├── CONTRIBUTING.md        # Contribution guidelines
└── LICENSE                # MIT license
```

## Building the Project

### Using Make (Recommended)
```bash
make build
```

### Manual Build Process
```bash
# Install dependencies with Conan
uv run conan install . --output-folder=build --build=missing

# Configure CMake
cmake --preset conan-release

# Build
cmake --build build/build/Release
```

### Available Targets
- `conancuda`: Simple "Hello World" C++ application
- `vectoradd`: CUDA vector addition example using cuda-api-wrappers

## Adding Dependencies

### Adding Conan Packages

1. **Edit `conanfile.py`**:
   ```python
   def requirements(self):
       self.requires("cuda-api-wrappers/0.8.0")
       self.requires("your-package/version")  # Add your package here
   ```

2. **Rebuild**:
   ```bash
   make build
   ```

3. **Use in CMakeLists.txt**:
   ```cmake
   find_package(your-package REQUIRED)
   target_link_libraries(your_target your-package::your-package)
   ```

### Popular CUDA/C++ Packages
- `cuda-api-wrappers`: Modern C++ CUDA API wrappers
- `thrust`: CUDA parallel algorithms library
- `cub`: CUDA primitives library
- `eigen`: Linear algebra library
- `opencv`: Computer vision library
- `boost`: C++ utilities

## Examples

### Vector Addition (vectorAdd.cu)
A complete CUDA example demonstrating:
- Memory allocation on GPU
- Kernel launch configuration
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

2. **Conan package not found**
   - Update Conan: `uv tool install --upgrade conan`
   - Check package availability: `conan search package-name`
   - Try different version: Update version in `conanfile.py`

3. **CMake configuration failed**
   - Clean build: `rm -rf build/`
   - Reinstall dependencies: `make build`
   - Check CMake version: `cmake --version`

4. **Dev container won't start**
   - Ensure Docker is running
   - Check NVIDIA Container Toolkit installation
   - Verify devcontainer.json syntax

### Getting Help
- Check [Issues](https://github.com/Hossein-Roshandel/conanCuda/issues) for known problems
- Create a new issue with your error message and system info
- See [CONTRIBUTING.md](CONTRIBUTING.md) for development guidelines

## Performance Tips

- **Memory Management**: Use unified memory for simpler code
- **Kernel Optimization**: Profile with `nvprof` or Nsight
- **Block Size**: Experiment with different block sizes (64, 128, 256, 512)
- **Memory Patterns**: Ensure coalesced memory access

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Contributing

Please read [CONTRIBUTING.md](CONTRIBUTING.md) for guidelines on contributing to this project.

## Acknowledgments

- NVIDIA CUDA samples and documentation
- cuda-api-wrappers library by Eyal Rozenberg
- Conan package manager community
- CMake development team