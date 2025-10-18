
# CUDA C++ Project with Conan and CMake
.PHONY: help build clean install run test debug info format lint memory-check docs pre-commit

# Default target
help:
	@echo "Available targets:"
	@echo "  build         - Install dependencies and build the project"
	@echo "  clean         - Clean build artifacts"
	@echo "  install       - Install dependencies only"
	@echo "  run           - Run the vector addition example"
	@echo "  test          - Run all tests"
	@echo "  debug         - Build in debug mode"
	@echo "  info          - Show build information"
	@echo "  format        - Format code with clang-format"
	@echo "  lint          - Run clang-tidy static analysis"
	@echo "  memory-check  - Run Valgrind and cuda-memcheck"
	@echo "  docs          - Generate Doxygen documentation"
	@echo "  pre-commit    - Install pre-commit hooks"
	@echo "  help          - Show this help message"

# Install dependencies and build (Release mode)
build:
	conan install . --output-folder=build --build=missing
	cmake --preset conan-release -DBUILD_TESTS=ON
	cmake --build build/build/Release

# Install dependencies only
install:
	conan install . --output-folder=build --build=missing

# Build in debug mode
debug:
	conan install . --output-folder=build --build=missing --settings=build_type=Debug
	cmake --preset conan-debug -DBUILD_TESTS=ON || cmake --preset conan-default -DBUILD_TESTS=ON
	cmake --build build/build/Debug || cmake --build build/build/Release

# Clean build artifacts
clean:
	rm -rf build/
	rm -rf out/
	rm -rf docs/html/
	rm -f *_report.txt
	rm -f compute_sanitizer_*.txt

# Run the vector addition example
run: build
	@echo "Running vector addition example..."
	./build/build/Release/vectoradd

# Run tests
test: build
	@echo "Running tests..."
	cd build/build/Release && ctest --verbose --output-on-failure

# Format code with clang-format
format:
	@echo "Formatting code..."
	find . -name "*.cpp" -o -name "*.hpp" -o -name "*.cu" -o -name "*.cuh" \
		| grep -v "build/" | xargs clang-format -i --style=file
	@echo "✓ Code formatted"

# Run static analysis
lint:
	@echo "Running static analysis..."
	./scripts/static_analysis.sh

# Run memory checks
memory-check: build
	@echo "Running memory analysis..."
	./scripts/memory_check.sh ./build/build/Release/vectoradd

# Generate documentation
docs:
	@echo "Generating documentation with Doxygen..."
	cmake --preset conan-release -DBUILD_DOCS=ON
	cmake --build build/build/Release --target docs
	@echo "✓ Documentation generated in docs/html/"
	@echo "Open docs/html/index.html in a browser"

# Install pre-commit hooks
pre-commit:
	@echo "Installing pre-commit hooks..."
	@if command -v pre-commit >/dev/null 2>&1; then \
		pre-commit install; \
		echo "✓ Pre-commit hooks installed"; \
	else \
		echo "⚠ pre-commit not found. Install with: pip install pre-commit"; \
	fi

# Show build information
info:
	@echo "=== Build Information ==="
	@echo "CMake version: $(shell cmake --version | head -n1)"
	@echo "Conan version: $(shell conan --version 2>/dev/null || echo 'Not installed')"
	@echo "CUDA version: $(shell nvcc --version 2>/dev/null | grep release || echo 'Not installed')"
	@echo "GPU info: $(shell nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null || echo 'No GPU detected')"
	@echo "Python version: $(shell python3 --version)"
	@echo "clang-format: $(shell clang-format --version 2>/dev/null | head -n1 || echo 'Not installed')"
	@echo "clang-tidy: $(shell clang-tidy --version 2>/dev/null | head -n1 || echo 'Not installed')"
	@echo "Doxygen: $(shell doxygen --version 2>/dev/null || echo 'Not installed')"
