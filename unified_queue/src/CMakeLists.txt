set(HEADERS
  "${PROJECT_SOURCE_DIR}/include/task_generator.h"
  "${PROJECT_SOURCE_DIR}/include/cpu_kernels.hpp"
  "${PROJECT_SOURCE_DIR}/include/gpu_kernels.h"
  "${PROJECT_SOURCE_DIR}/include/parser.hpp"
)

add_library(gpu_kernels 
  task_generator.cu
  cpu_kernels.cpp 
  gpu_kernels.cu 
  parser.cpp 
  ${HEADERS})

target_include_directories(gpu_kernels PUBLIC ../include)

target_compile_features(gpu_kernels PUBLIC cxx_std_11)

add_executable(main main.cpp)
target_link_libraries(main PRIVATE gpu_kernels)
