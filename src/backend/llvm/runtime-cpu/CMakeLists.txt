add_athena_library(runtime-llvm-cpu SHARED
        add.cpp
        allocate.cpp
        fill.cpp
        matrix.cpp
        mse.cpp
        ${WRAPPER_CPP})
add_dependencies(runtime-llvm-cpu wrapper_cpp)

find_package(BLAS)

if (BLAS_FOUND)
    if (APPLE)
        target_compile_definitions(runtime-llvm-cpu PRIVATE ATHENA_APPLE_ACCELERATE=1)
    endif ()
    message(STATUS ${BLAS_LIBRARIES})
    find_path(BLAS_INCLUDE_DIRS cblas.h
            /usr/include
            /usr/local/include
            $ENV{BLAS_HOME}/include)
    target_link_libraries(runtime-llvm-cpu PRIVATE ${BLAS_LIBRARIES})
    if (BLAS_INCLUDE_DIRS)
        target_include_directories(runtime-llvm-cpu PRIVATE ${BLAS_INCLUDE_DIRS})
    endif ()
else ()
    include(ExternalProject)
    ExternalProject_Add(OpenBlasDownload
            PREFIX ${CMAKE_BINARY_DIR}/_deps/OpenBLAS-0.3.7
            GIT_REPOSITORY https://github.com/xianyi/OpenBLAS.git
            GIT_TAG "v0.3.7"
            CONFIGURE_COMMAND ""
            BUILD_IN_SOURCE 1
            BUILD_COMMAND make
            INSTALL_COMMAND ""
            UPDATE_COMMAND ""
            )
    set(OpenBlas_DIR ${CMAKE_BINARY_DIR}/_deps/OpenBLAS-0.3.7/src/OpenBlasDownload)
    target_link_directories(runtime-llvm-cpu PRIVATE ${OpenBlas_DIR})
    target_link_libraries(runtime-llvm-cpu PRIVATE openblas)
    target_include_directories(runtime-llvm-cpu PRIVATE ${OpenBlas_DIR})
    add_dependencies(runtime-llvm-cpu OpenBlasDownload)
endif ()

set(WEAK_FLAG)
if (APPLE)
    set(WEAK_FLAG -Wl,-weak_library)
endif ()

target_link_libraries(runtime-llvm-cpu PRIVATE ${WEAK_FLAG} athena-core)

install(TARGETS runtime-llvm-cpu EXPORT AthenaConfig
        RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR}
        COMPONENT Athena_Runtime
        LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
        COMPONENT Athena_Runtime
        NAMELINK_COMPONENT Athena_Development
        ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR}
        COMPONENT Athena_Development
        )
