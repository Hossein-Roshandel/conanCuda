# CUDA C++ Development Container with Conan and CMake
FROM nvidia/cuda:12.9.1-devel-ubuntu22.04

# Avoid timezone prompt during installation
ENV DEBIAN_FRONTEND=noninteractive
ENV TZ=UTC

# Install system dependencies
RUN apt-get update && apt-get install -y \
    # Build essentials
    build-essential \
    cmake \
    ninja-build \
    pkg-config \
    # Development tools
    git \
    curl \
    wget \
    unzip \
    vim \
    nano \
    # Python and package management
    python3 \
    python3-pip \
    python3-venv \
    # Utilities
    sudo \
    ca-certificates \
    gnupg \
    lsb-release \
    software-properties-common \
    # Libraries that might be needed
    libssl-dev \
    zlib1g-dev \
    && rm -rf /var/lib/apt/lists/*

RUN wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2204/x86_64/cuda-keyring_1.1-1_all.deb && \
    dpkg -i cuda-keyring_1.1-1_all.deb && \
    apt-get update && apt-get install -y \
    cuda-drivers \
    && rm cuda-keyring_1.1-1_all.deb \
    && rm -rf /var/lib/apt/lists/*

# Install latest CMake (if needed)
RUN CMAKE_VERSION="3.28.1" && \
    wget -O cmake.tar.gz "https://github.com/Kitware/CMake/releases/download/v${CMAKE_VERSION}/cmake-${CMAKE_VERSION}-linux-x86_64.tar.gz" && \
    tar -xzf cmake.tar.gz --strip-components=1 -C /usr/local && \
    rm cmake.tar.gz

# install uv
COPY --from=astral/uv:latest /uv /uvx /bin/

# Create non-root user
RUN useradd -m -s /bin/bash -G sudo vscode && \
    echo "vscode ALL=(ALL) NOPASSWD:ALL" >> /etc/sudoers

# Switch to vscode user and install user-level tools
USER vscode
WORKDIR /home/vscode

# Install Conan and pre-commit using uv
RUN uv tool install conan
RUN uv tool install pre-commit --with pre-commit-uv

WORKDIR /workspace

# Set environment variables for CUDA
ENV CUDA_HOME=/usr/local/cuda
ENV PATH=${CUDA_HOME}/bin:${PATH}
ENV LD_LIBRARY_PATH=${CUDA_HOME}/lib64:${LD_LIBRARY_PATH}

CMD ["/bin/bash"]
