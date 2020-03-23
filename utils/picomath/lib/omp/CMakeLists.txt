set(OMP_SRC
        src/blas.cpp
        )

add_library(picomath_omp STATIC ${OMP_SRC})
athena_disable_exceptions(picomath_omp)
athena_disable_rtti(picomath_omp)
target_include_directories(picomath_omp PUBLIC ${PICOMATH_DIR}/include)
target_include_directories(picomath_omp PRIVATE ${PICOMATH_SRC_DIR}/omp/src)
target_include_directories(picomath_omp PRIVATE ${PICOMATH_SRC_DIR}/omp/include)
target_include_directories(picomath_omp PRIVATE ${PICOMATH_SRC_DIR}/)

find_package(OpenMP)

if (OpenMP_CXX_Found)
    target_link_libraries(picomath_omp PRIVATE OpenMP::OpenMP_CXX)
endif ()

if (CMAKE_BUILD_TYPE MATCHES "Release")
    target_compile_options(picomath_omp PRIVATE -O3)
    check_cxx_compiler_flag(-flto=thin HAS_THIN_LTO)
    check_cxx_compiler_flag("-O3 -mllvm -polly" HAS_POLLY)

    if (HAS_THIN_LTO)
        target_compile_options(picomath_omp PRIVATE -flto=thin)
    endif ()
    if (HAS_POLLY)
        target_compile_options(picomath_omp PRIVATE -mllvm -polly)
    endif ()
endif ()