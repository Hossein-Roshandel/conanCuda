# GPU Setup for Dev Container

## Overview

The dev container is configured to work both **with** and **without** GPU access. By default, GPU access is **disabled** to ensure compatibility across all systems.

## Current Status

✅ **Container will start without GPU** - You can develop and build CUDA code  
❌ **CUDA programs won't run** - You'll see "No CUDA devices" error when executing

## Enabling GPU Access

To run CUDA programs in the dev container, you need to:

### Step 1: Install NVIDIA Container Toolkit

#### On Ubuntu/Debian (WSL2):

```bash
# Add NVIDIA repository
distribution=$(. /etc/os-release;echo $ID$VERSION_ID)
curl -fsSL https://nvidia.github.io/libnvidia-container/gpgkey | sudo gpg --dearmor -o /usr/share/keyrings/nvidia-container-toolkit-keyring.gpg
curl -s -L https://nvidia.github.io/libnvidia-container/$distribution/libnvidia-container.list | \
    sed 's#deb https://#deb [signed-by=/usr/share/keyrings/nvidia-container-toolkit-keyring.gpg] https://#g' | \
    sudo tee /etc/apt/sources.list.d/nvidia-container-toolkit.list

# Install the toolkit
sudo apt-get update
sudo apt-get install -y nvidia-container-toolkit

# Configure Docker to use NVIDIA runtime
sudo nvidia-ctk runtime configure --runtime=docker
sudo systemctl restart docker

# Verify installation
docker run --rm --runtime=nvidia --gpus all nvidia/cuda:12.2.0-base-ubuntu22.04 nvidia-smi
```

#### For Windows with Docker Desktop + WSL2:

1. Ensure you have:
   - Windows 11 or Windows 10 21H2+
   - NVIDIA GPU drivers (latest)
   - Docker Desktop with WSL2 backend enabled

2. Install NVIDIA Container Toolkit in your WSL2 distribution (follow Ubuntu steps above)

3. In Docker Desktop settings:
   - Go to Settings → Resources → WSL Integration
   - Enable integration with your Ubuntu distribution

### Step 2: Enable GPU in docker-compose.yml

Edit `.devcontainer/docker-compose.yml` and uncomment the GPU section:

```yaml
services:
  dev:
    build:
      context: .
      dockerfile: Dockerfile
    volumes:
      - ..:/workspace:cached
    working_dir: /workspace
    command: sleep infinity
    environment:
      - NVIDIA_VISIBLE_DEVICES=all
      - NVIDIA_DRIVER_CAPABILITIES=compute,utility
    # Uncomment these lines for GPU access:
    runtime: nvidia
    deploy:
      resources:
        reservations:
          devices:
            - driver: nvidia
              count: all
              capabilities: [gpu]
```

### Step 3: Rebuild Dev Container

In VS Code:
1. Press `Ctrl+Shift+P` (or `Cmd+Shift+P` on Mac)
2. Select "Dev Containers: Rebuild Container"
3. Wait for the container to rebuild

### Step 4: Verify GPU Access

Once the container is running:

```bash
# Check if GPU is visible
nvidia-smi

# Run the CUDA example
make run
```

## Troubleshooting

### "nvidia-smi: command not found"
- NVIDIA drivers are not installed on your host system
- Install the latest NVIDIA drivers for your GPU

### "docker: Error response from daemon: unknown or invalid runtime name: nvidia"
- NVIDIA Container Toolkit is not installed
- Follow Step 1 above to install it
- Make sure you restarted Docker after installation

### "No CUDA devices on this system"
- GPU passthrough is not configured in docker-compose.yml
- Uncomment the GPU section as shown in Step 2
- Rebuild the container

### WSL2 Specific Issues

If `nvidia-smi` works in WSL2 but not in Docker:
```bash
# Check WSL2 GPU support
nvidia-smi

# Check Docker can access GPU
docker run --rm --runtime=nvidia --gpus all nvidia/cuda:12.2.0-base-ubuntu22.04 nvidia-smi

# If the above fails, reconfigure Docker
sudo nvidia-ctk runtime configure --runtime=docker
sudo systemctl restart docker
```

## Development Without GPU

You can still:
- ✅ Write CUDA code with full IntelliSense
- ✅ Build CUDA programs (`make build`)
- ✅ Edit and manage dependencies
- ✅ Use all development tools (CMake, Conan, etc.)
- ❌ Cannot execute CUDA programs (will fail at runtime)

To test your code on GPU, you can:
1. Build in the dev container
2. Copy the executable to your host system
3. Run it on the host (if you have CUDA installed locally)

Or better yet, set up GPU access following the steps above!

## Additional Resources

- [NVIDIA Container Toolkit Installation](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html)
- [Docker GPU Support](https://docs.docker.com/config/containers/resource_constraints/#gpu)
- [WSL2 GPU Support](https://learn.microsoft.com/en-us/windows/wsl/tutorials/gpu-compute)
