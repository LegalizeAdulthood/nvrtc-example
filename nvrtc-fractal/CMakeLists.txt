add_executable(nvrtc-fractal
    main.cpp
    stubs.cpp
    Complex.cuh
    Fractal.h
    Fractal.cu
)
target_link_libraries(nvrtc-fractal PUBLIC OptiXToolkit::Gui CUDA::nvrtc)

# Copy shared libraries that the built executable depends on.
add_custom_command( TARGET nvrtc-fractal POST_BUILD
    COMMAND ${CMAKE_COMMAND} -E copy $<TARGET_RUNTIME_DLLS:nvrtc-fractal> $<TARGET_FILE_DIR:nvrtc-fractal>
    COMMAND_EXPAND_LISTS )
