option(PICOMATH_ENABLE_OMP ON "Enable OpenMP engine for PicoMath library")

set(PICOMATH_DIR ${CMAKE_CURRENT_SOURCE_DIR})
add_subdirectory(lib)

add_subdirectory(test)