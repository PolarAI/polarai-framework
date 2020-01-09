set(ATH_BACKEND_LLVM "AthenaBackendLLVM" CACHE STRING "" FORCE)

find_package(LLVM CONFIG)
find_package(PythonInterp REQUIRED)

message(STATUS "Found LLVM ${LLVM_PACKAGE_VERSION}")
message(STATUS "Using LLVMConfig.cmake in: ${LLVM_DIR}")

set(WRAPPER_CPP ${CMAKE_CURRENT_BINARY_DIR}/wrapper.cpp)
set(GENERATOR_CPP ${CMAKE_CURRENT_BINARY_DIR}/generator.cpp)

file(WRITE ${WRAPPER_CPP} "")
file(WRITE ${GENERATOR_CPP} "")

add_custom_target(wrapper_cpp
        COMMAND ${PYTHON_EXECUTABLE} ${CMAKE_CURRENT_SOURCE_DIR}/runtime/tablegen.py
        ${CMAKE_CURRENT_SOURCE_DIR}/runtime/builtins.td ${WRAPPER_CPP} wrapper
        )

add_custom_target(generator_cpp
        COMMAND ${PYTHON_EXECUTABLE}
        ${CMAKE_CURRENT_SOURCE_DIR}/runtime/tablegen.py
        ${CMAKE_CURRENT_SOURCE_DIR}/runtime/builtins.td ${GENERATOR_CPP} driver
        )

add_subdirectory(runtime/cpu)
add_subdirectory(runtime/driver)
add_subdirectory(runtime/legacy_driver)

file(GLOB codegen_src "codegen/*.cpp")

add_athena_library(${ATH_BACKEND_LLVM} STATIC
        ATH_BACKEND_LLVM # export name
        backend/llvm/llvm_export.h
        LLVMExecutor.cpp
        LLVMGenerator.cpp
        jit/AthenaJIT.cpp
        jit/MergeLayer.cpp
        LLVMTrivialAllocator.cpp
        GraphPartitionPlanner.cpp
        utils.cpp
        ${codegen_src})

target_include_directories(${ATH_BACKEND_LLVM} PRIVATE ${LLVM_INCLUDE_DIRS})
target_compile_definitions(${ATH_BACKEND_LLVM} PRIVATE ${LLVM_DEFINITIONS})

llvm_map_components_to_libnames(llvm_libs
        core
        irreader
        support
        transformutils
        passes
        codegen
        orcjit
        x86asmparser
        x86codegen
        lto)

add_dependencies(${ATH_BACKEND_LLVM} ${ATH_RT_LLVM_CPU})
target_link_libraries(${ATH_BACKEND_LLVM}
        PRIVATE ${llvm_libs}
        PRIVATE ${ATH_RT_LLVM_DRIVER})

find_package(MLIR COMPONENTS IR)

if (MLIR_LIBS AND MLIR_TBLGEN AND MLIR_OPT)
    add_subdirectory(mlir)
endif ()
