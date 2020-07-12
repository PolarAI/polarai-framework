set(ATH_BACKEND_LLVM "AthenaBackendLLVM" CACHE STRING "" FORCE)

if(POLICY CMP0068)
    cmake_policy(SET CMP0068 NEW)
    set(CMAKE_BUILD_WITH_INSTALL_NAME_DIR ON)
endif()

if(POLICY CMP0075)
    cmake_policy(SET CMP0075 NEW)
endif()

if(POLICY CMP0077)
    cmake_policy(SET CMP0077 NEW)
endif()

find_package(LLVM REQUIRED CONFIG)
find_package(MLIR REQUIRED CONFIG)
find_package(Python3 COMPONENTS Interpreter)

list(APPEND CMAKE_MODULE_PATH "${MLIR_CMAKE_DIR}")
list(APPEND CMAKE_MODULE_PATH "${LLVM_CMAKE_DIR}")
include(TableGen)
include(AddLLVM)
include(AddMLIR)

# FIXME Ninja doesn't work with this
# include(HandleLLVMOptions)

message(STATUS "Found LLVM ${LLVM_PACKAGE_VERSION}")
message(STATUS "Using LLVMConfig.cmake in: ${LLVM_DIR}")

add_subdirectory(jit)

add_athena_library(${ATH_BACKEND_LLVM} SHARED
        ATH_BACKEND_LLVM # export name
        backend/llvm/llvm_export.h
        LLVMExecutor.cpp
        GraphPartitionPlanner.cpp
        allocators/LayerAllocator.cpp
        CodeGen.cpp)

target_include_directories(${ATH_BACKEND_LLVM} PRIVATE 
                          ${LLVM_INCLUDE_DIRS}
                          ${CMAKE_CURRENT_BINARY_DIR})
target_compile_definitions(${ATH_BACKEND_LLVM} PRIVATE ${LLVM_DEFINITIONS})

llvm_map_components_to_libnames(llvm_libs
        core
        irreader
        support
        transformutils
        passes
        codegen
        orcjit
        AllTargetsCodeGens
        AllTargetsAsmParsers
        AllTargetsDescs
        AllTargetsInfos
        lto)

target_link_libraries(${ATH_BACKEND_LLVM} PRIVATE
        ${llvm_libs}
        ${AthenaJitTarget}
        MLIRIR
        MLIRStandardOps
        MLIRPolarGraph
        ${ATH_RT_LLVM_SUPPORT_OBJ}
        ${ATH_RT_LLVM_DRIVER}
        )
target_include_directories(${ATH_BACKEND_LLVM} PRIVATE
        ${PROJECT_SOURCE_DIR}/utils/dialects/include
        ${PROJECT_BINARY_DIR}/utils/dialects/include)
