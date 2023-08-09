cmake_minimum_required(VERSION 3.22 FATAL_ERROR)

project(nvrtc-fractal)

find_package(CUDAToolkit)

set_property(GLOBAL PROPERTY USE_FOLDERS ON)

add_subdirectory(nvrtc-fractal)
