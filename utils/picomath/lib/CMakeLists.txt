set(PICOMATH_SRC_DIR ${CMAKE_CURRENT_SOURCE_DIR})

message(STATUS "Building picomath with OpenMP")
add_subdirectory(omp)

if (PICOMATH_ENABLE_SYCL)
    add_subdirectory(sycl)
endif()