cmake_minimum_required(VERSION 3.13)

project(picomath CXX)

option(PICOMATH_ENABLE_OMP ON "Enable OpenMP engine")
option(PICOMATH_ENABLE_SYCL OFF "Enable SYCL engine")

set(PICOMATH_DIR ${CMAKE_CURRENT_SOURCE_DIR})
add_subdirectory(lib)

add_subdirectory(test)