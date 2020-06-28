option(ATHENA_ENABLE_EXPERIMENTAL_SYCL "Enable experimental SYCL Runtime" ON)
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

add_subdirectory(runtime/support)
add_subdirectory(jit)

set(ATH_HAS_OPENCL "false" CACHE STRING "" FORCE)

add_athena_library(${ATH_BACKEND_LLVM} STATIC
        ATH_BACKEND_LLVM # export name
        backend/llvm/llvm_export.h
        LLVMExecutor.cpp
        GraphPartitionPlanner.cpp
        allocators/LayerAllocator.cpp
        ImageManager.cpp
        CodeGen.cpp)

if (ATHENA_ENABLE_EXPERIMENTAL_SYCL)
  set(ATH_HAS_SYCL "true" CACHE STRING "" FORCE)
  add_subdirectory(runtime/sycl)
  add_dependencies(${ATH_BACKEND_LLVM} ${ATH_RT_LLVM_SYCL})
else()
  set(ATH_HAS_SYCL "false" CACHE STRING "" FORCE)
endif()

find_package(CUDAToolkit)
if (CUDAToolkit_FOUND)
  add_subdirectory(runtime/cuda)
  add_dependencies(${ATH_BACKEND_LLVM} ${POLAR_RT_LLVM_CUDA})
else()
  set(POLAR_HAS_CUDA "false" CACHE STRING "" FORCE)
endif()

if (UNIX AND NOT APPLE)
  if (NOT DEFINED ROCM_PATH)
    if (NOT DEFINED ENV{ROCM_PATH})
      set(ROCM_PATH "/opt/rocm" CACHE PATH "Path to which ROCm has been installed")
    else()
      set(ROCM_PATH $ENV{ROCM_PATH} CACHE PATH "Path to which ROCm has been installed")
    endif()
    set(HIP_PATH "${ROCM_PATH}/hip" CACHE PATH " Path to which HIP has been installed")
  endif()
  set(CMAKE_MODULE_PATH "${HIP_PATH}/cmake" ${CMAKE_MODULE_PATH})
  find_package(HIP)
  #if (HIP_FOUND)
  #  add_subdirectory(runtime/hip)
  #  add_dependencies(${ATH_BACKEND_LLVM} ${POLAR_RT_LLVM_HIP})
  #else()
    set(POLAR_HAS_HIP "false" CACHE STRING "" FORCE)
  #endif()
endif()

find_package(Vulkan)
if (Vulkan_FOUND)
  add_subdirectory(runtime/vulkan)
  add_dependencies(${ATH_BACKEND_LLVM} ${POLAR_RT_LLVM_VULKAN})
else()
  set(POLAR_HAS_VULKAN "false" CACHE STRING "" FORCE)
endif()
  set(POLAR_HAS_VULKAN "false" CACHE STRING "" FORCE)

add_subdirectory(runtime/driver)

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
