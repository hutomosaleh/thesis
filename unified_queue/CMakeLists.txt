cmake_minimum_required(VERSION 3.2 FATAL_ERROR)

project(
  tpch
  VERSION 0.1
  DESCRIPTION "Task Queue"
  LANGUAGES CXX CUDA
)

set(CMAKE_CXX_FLAGS "-Wall -Wextra")
set(CMAKE_CXX_FLAGS_DEBUG "-g")
set(CMAKE_CXX_FLAGS_RELEASE "-O2")
set(CMAKE_EXPORT_COMPILE_COMMANDS ON CACHE INTERNAL "")

add_subdirectory(src)

