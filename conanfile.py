from conan import ConanFile
from conan.tools.cmake import cmake_layout


class ExampleRecipe(ConanFile):
    settings = "os", "compiler", "build_type", "arch"
    generators = "CMakeDeps", "CMakeToolchain"

    def requirements(self):
        self.requires("cuda-api-wrappers/0.8.0")

    def build_requirements(self):
        """Build-time dependencies for testing and development"""
        self.test_requires("gtest/1.14.0")

    def layout(self):
        cmake_layout(self)
