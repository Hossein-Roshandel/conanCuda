cmake_minimum_required(VERSION 3.5.0)
project(conancuda VERSION 0.1.0 LANGUAGES C CXX CUDA)
# include(CheckLanguage)
# check_language(CUDA)

add_executable(conancuda main.cpp)
add_executable(vectoradd vectorAdd.cu)

include(CTest)
enable_testing()

set(CPACK_PROJECT_NAME ${PROJECT_NAME})
set(CPACK_PROJECT_VERSION ${PROJECT_VERSION})
include(CPack)


find_package(cuda-api-wrappers REQUIRED)


target_link_libraries(vectoradd cuda-api-wrappers::runtime-and-driver cuda)