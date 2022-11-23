cmake_minimum_required(VERSION 3.2 FATAL_ERROR)

project(
  tpch
  VERSION 0.1
  DESCRIPTION "Thesis Exercise"
  LANGUAGES CXX CUDA
)

set(CMAKE_EXPORT_COMPILE_COMMANDS ON CACHE INTERNAL "")

add_subdirectory(src)
add_subdirectory(apps)

