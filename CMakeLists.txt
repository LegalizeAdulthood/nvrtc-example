cmake_minimum_required(VERSION 3.22 FATAL_ERROR)

project(nvrtc-fractal LANGUAGES CUDA CXX)

find_package(CUDAToolkit)

set_property(GLOBAL PROPERTY USE_FOLDERS ON)
set(OTK_BUILD_EXAMPLES ON)

add_subdirectory(nvrtc-fractal)
add_subdirectory(optix-toolkit)
