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

message(STATUS "Found LLVM ${LLVM_PACKAGE_VERSION}")
message(STATUS "Using LLVMConfig.cmake in: ${LLVM_DIR}")

add_subdirectory(jit)

add_polar_library(polar_backend_generic SHARED
        Executor.cpp
        ExecutorImpl.cpp
        GraphPartitionPlanner.cpp
        allocators/LayerAllocator.cpp
        CodeGen.cpp
        ScriptOperationInternal.cpp)

add_library(PolarAI::generic_backend ALIAS polar_backend_generic)

target_include_directories(polar_backend_generic PRIVATE 
                          ${LLVM_INCLUDE_DIRS}
                          ${CMAKE_CURRENT_BINARY_DIR})
target_compile_definitions(polar_backend_generic PRIVATE ${LLVM_DEFINITIONS})

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

target_link_libraries(polar_backend_generic PRIVATE
        ${llvm_libs}
        ${PolarJitTarget}
        MLIRIR
        MLIRStandardOps
        MLIRPolarGraph
        PolarAI::generic_backend_support
        PolarAI::generic_backend_driver
        PUBLIC
        PolarAI::framework
        PolarAI::utils
        )
target_include_directories(polar_backend_generic PRIVATE
        ${PROJECT_SOURCE_DIR}/utils/dialects/include
        ${PROJECT_BINARY_DIR}/utils/dialects/include)

install(TARGETS polar_backend_generic EXPORT PolarAIConfig
        RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR}
        COMPONENT PolarAI_Runtime
        LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
        COMPONENT PolarAI_Runtime
        NAMELINK_COMPONENT PolarAI_Development
        ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR}
        COMPONENT PolarAI_Development
        )
