add_executable(q6 q6.cu)
target_compile_features(q6 PRIVATE cxx_std_17)
target_link_libraries(q6 PRIVATE gpu_kernels)

