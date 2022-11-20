set(HEADERS "${PROJECT_SOURCE_DIR}/include/gpu_kernels.h")

add_library(gpu_kernels gpu_kernels.cu ${HEADERS})

target_include_directories(gpu_kernels PUBLIC ../include)

target_compile_features(gpu_kernels PUBLIC cxx_std_11)
