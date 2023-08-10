function(copy_dependencies target)
    # Copy shared libraries that the built executable depends on.
    add_custom_command( TARGET ${target} POST_BUILD
        COMMAND ${CMAKE_COMMAND} -E copy $<TARGET_RUNTIME_DLLS:${target}> $<TARGET_FILE_DIR:${target}>
        COMMAND_EXPAND_LISTS )
endfunction()

add_executable(nvrtc-fractal
    main.cpp
    stubs.cpp
    Complex.cuh
    Formula.cuh
    Fractal.h
    Fractal.cu
    Iterate.cuh
    Iterate.cu
)
set_target_properties(nvrtc-fractal PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
target_link_libraries(nvrtc-fractal PUBLIC OptiXToolkit::Error OptiXToolkit::Gui)
copy_dependencies(nvrtc-fractal)

configure_file(SourceDir.cpp.in SourceDir.cpp)
add_executable(nvrtc-fractal-dynamic
    main-dynamic.cpp
    stubs.cpp
    Dynamic.h
    Dynamic.cpp
    SourceDir.h
    SourceDir.cpp.in
    ${CMAKE_CURRENT_BINARY_DIR}/SourceDir.cpp
    nvrtcErrorCheck.h
    nvJitLinkErrorCheck.h
)
target_include_directories(nvrtc-fractal-dynamic PUBLIC .)
target_link_libraries(nvrtc-fractal-dynamic PUBLIC OptiXToolkit::Error OptiXToolkit::Gui CUDA::nvrtc CUDA::nvJitLink)
copy_dependencies(nvrtc-fractal-dynamic)
