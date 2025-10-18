
# CUDA C++ Project with Conan and CMake
.PHONY: help build clean install run test debug info

# Default target
help:
	@echo "Available targets:"
	@echo "  build    - Install dependencies and build the project"
	@echo "  clean    - Clean build artifacts"
	@echo "  install  - Install dependencies only"
	@echo "  run      - Run the vector addition example"
	@echo "  test     - Run all tests"
	@echo "  debug    - Build in debug mode"
	@echo "  info     - Show build information"
	@echo "  help     - Show this help message"

# Install dependencies and build (Release mode)
build:
	uv run conan install . --output-folder=build --build=missing
	cmake --preset conan-release
	cmake --build build/build/Release

# Install dependencies only
install:
	uv run conan install . --output-folder=build --build=missing

# Build in debug mode
debug:
	uv run conan install . --output-folder=build --build=missing --settings=build_type=Debug
	cmake --preset conan-debug || cmake --preset conan-default
	cmake --build build/build/Debug || cmake --build build/build/Release

# Clean build artifacts
clean:
	rm -rf build/
	rm -rf out/

# Run the vector addition example
run: build
	@echo "Running vector addition example..."
	./build/build/Release/vectoradd

# Run tests
test: build
	@echo "Running tests..."
	cd build/build/Release && ctest --verbose

# Show build information
info:
	@echo "=== Build Information ==="
	@echo "CMake version: $(shell cmake --version | head -n1)"
	@echo "Conan version: $(shell uv run conan --version 2>/dev/null || echo 'Not installed')"
	@echo "CUDA version: $(shell nvcc --version 2>/dev/null | grep release || echo 'Not installed')"
	@echo "GPU info: $(shell nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null || echo 'No GPU detected')"
	@echo "Python version: $(shell python3 --version)"
	@echo "uv version: $(shell uv --version 2>/dev/null || echo 'Not installed')"