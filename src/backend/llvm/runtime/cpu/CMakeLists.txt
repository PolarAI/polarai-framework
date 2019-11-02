set(ATH_RT_LLVM_CPU "AthenaLLVMCpuRuntime" CACHE STRING "" FORCE)

add_athena_library(${ATH_RT_LLVM_CPU} SHARED
        ATH_RT_LLVM # export name
        backend/llvm/runtime/runtime_export.h # export file name
        add.cpp
        allocate.cpp
        fill.cpp
        matrix.cpp
        mse.cpp
        ${WRAPPER_CPP})
add_dependencies(${ATH_RT_LLVM_CPU} wrapper_cpp)
target_compile_options(${ATH_RT_LLVM_CPU} PRIVATE -fvisibility=default)

find_package(BLAS)

if (BLAS_FOUND)
    if (APPLE)
        target_compile_definitions(${ATH_RT_LLVM_CPU} PRIVATE ATHENA_APPLE_ACCELERATE=1)
    endif ()
    message(STATUS ${BLAS_LIBRARIES})
    find_path(BLAS_INCLUDE_DIRS cblas.h
            /usr/include
            /usr/local/include
            $ENV{BLAS_HOME}/include)
    target_link_libraries(${ATH_RT_LLVM_CPU} PRIVATE ${BLAS_LIBRARIES})
    if (BLAS_INCLUDE_DIRS)
        target_include_directories(${ATH_RT_LLVM_CPU} PRIVATE ${BLAS_INCLUDE_DIRS})
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
    target_link_directories(${ATH_RT_LLVM_CPU} PRIVATE ${OpenBlas_DIR})
    target_link_libraries(${ATH_RT_LLVM_CPU} PRIVATE openblas)
    target_include_directories(${ATH_RT_LLVM_CPU} PRIVATE ${OpenBlas_DIR})
    add_dependencies(${ATH_RT_LLVM_CPU} OpenBlasDownload)
endif ()

set(WEAK_FLAG)
set(NOWEAK_FLAG)
if (APPLE)
    set(WEAK_FLAG -Wl,-weak_library)
elseif (UNIX)
    set(WEAK_FLAG -Wl,--as-needed)
    set(NOWEAK_FLAG -Wl,--no-as-needed)
endif ()

target_link_libraries(${ATH_RT_LLVM_CPU} PRIVATE ${WEAK_FLAG} ${ATH_CORE} ${NOWEAK_FLAG})
target_compile_definitions(${ATH_RT_LLVM_CPU} PRIVATE -D${ATH_CORE}_WEAK)

install(TARGETS ${ATH_RT_LLVM_CPU} EXPORT AthenaConfig
        RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR}
        COMPONENT Athena_Runtime
        LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
        COMPONENT Athena_Runtime
        NAMELINK_COMPONENT Athena_Development
        ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR}
        COMPONENT Athena_Development
        )
