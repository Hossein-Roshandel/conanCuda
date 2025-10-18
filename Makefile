
build:
	uv run conan install . --output-folder=build --build=missing
	cmake --preset conan-release