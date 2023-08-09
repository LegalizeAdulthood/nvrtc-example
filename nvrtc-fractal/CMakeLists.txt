add_executable(nvrtc-fractal main.cpp)
target_link_libraries(nvrtc-fractal PUBLIC CUDA::nvrtc)
