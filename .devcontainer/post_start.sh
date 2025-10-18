#!/bin/bash
# Post-start script for CUDA C++ development container
# This script runs after the container starts

echo "=========================================="
echo "Container started"
echo "=========================================="

# Check GPU availability
echo "Checking GPU status..."
if command -v nvidia-smi &> /dev/null; then
    if nvidia-smi --query-gpu=name,driver_version --format=csv,noheader 2>/dev/null; then
        echo "✅ GPU is available and accessible"
    else
        echo "⚠️  nvidia-smi found but GPU not accessible"
        echo "   This may be normal if running without GPU passthrough"
    fi
else
    echo "⚠️  nvidia-smi not found - GPU support not available"
fi

echo "=========================================="
echo "Ready to work!"
echo "=========================================="
