set(SYCL_SRC
        src/blas.cpp
        )

add_library(picomath_sycl SHARED ${SYCL_SRC})
target_include_directories(picomath_sycl PUBLIC ${PICOMATH_DIR}/include)
target_include_directories(picomath_sycl PRIVATE ${PICOMATH_SRC_DIR}/sycl/include)
target_include_directories(picomath_sycl PRIVATE ${PICOMATH_SRC_DIR}/)

check_cxx_compiler_flag(-fsycl HAS_FSYCL)
if (HAS_FSYCL)
    target_compile_options(picomath_sycl PRIVATE -fsycl)
    target_link_options(picomath_sycl PRIVATE -fsycl)
endif()