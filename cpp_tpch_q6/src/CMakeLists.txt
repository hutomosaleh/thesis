set(HEADERS "${PROJECT_SOURCE_DIR}/include/gpu_kernels.h" "${PROJECT_SOURCE_DIR}/include/parser.hpp")

add_library(gpu_kernels gpu_kernels.cu parser.cpp ${HEADERS})

target_include_directories(gpu_kernels PUBLIC ../include)

target_compile_features(gpu_kernels PUBLIC cxx_std_11)
