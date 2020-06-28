set(POLAR_RT_LLVM_HIP "PolarLLVMHIPRT" CACHE STRING "" FORCE)
set(POLAR_HAS_HIP "true" CACHE STRING "" FORCE)

find_library(ROCM_RUNTIME_LIBRARY hip_hcc
               PATHS "${HIP_PATH}/lib")
if (NOT ROCM_RUNTIME_LIBRARY)
  message(SEND_ERROR "Could not locate ROCm HIP runtime library")
else()
  message(STATUS "ROCm HIP runtime lib: ${ROCM_RUNTIME_LIBRARY}")
endif()

add_athena_library(${POLAR_RT_LLVM_HIP} MODULE
        ATH_RT_LLVM # export name
        backend/llvm/runtime/runtime_export.h # export file name

        api.cpp
        HIPContext.cpp
        HIPDevice.cpp
  )

target_link_libraries(${POLAR_RT_LLVM_HIP} PRIVATE ${ROCM_RUNTIME_LIBRARY})
target_compile_definitions(${POLAR_RT_LLVM_HIP} PRIVATE __HIP_PLATFORM_HCC__)
target_include_directories(${POLAR_RT_LLVM_HIP} SYSTEM PRIVATE
  "${HIP_PATH}/../include"
  "${HIP_PATH}/include")
