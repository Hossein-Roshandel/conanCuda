#!/bin/bash
# Post-create script for CUDA C++ development container
# This script runs after the container is created and workspace is mounted

set -e  # Exit on error

echo "=========================================="
echo "Starting post-create setup..."
echo "=========================================="

# Detect and configure Conan profile
echo "📦 Configuring Conan profile..."
conan profile detect --force
echo "✅ Conan profile configured successfully"

# Display Conan profile information
echo ""
echo "Conan profile details:"
conan profile show

# Display environment information
echo ""
echo "=========================================="
echo "Environment Information:"
echo "=========================================="
echo "CMake version: $(cmake --version | head -n1)"
echo "Conan version: $(conan --version)"
echo "CUDA version: $(nvcc --version | grep release | awk '{print $5}' | sed 's/,//')"
echo "Python version: $(python3 --version)"
echo "GCC version: $(gcc --version | head -n1)"

# Check GPU availability
echo ""
echo "=========================================="
echo "GPU Information:"
echo "=========================================="
if command -v nvidia-smi &> /dev/null; then
    nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv,noheader || echo "⚠️  GPU not available or NVIDIA drivers not installed"
else
    echo "⚠️  nvidia-smi not found - GPU support may not be available"
fi

echo ""
echo "=========================================="
echo "✅ Container ready for CUDA C++ development!"
echo "=========================================="
echo ""
echo "Quick start commands:"
echo "  make build    - Build the project"
echo "  make run      - Run the vector addition example"
echo "  make test     - Run tests"
echo "  make info     - Show build information"
echo ""
