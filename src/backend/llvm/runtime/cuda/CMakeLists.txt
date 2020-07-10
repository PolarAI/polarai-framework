set(POLAR_RT_LLVM_CUDA "PolarLLVMCudaRT" CACHE STRING "" FORCE)
set(POLAR_HAS_CUDA "true" CACHE STRING "" FORCE)

add_athena_library(${POLAR_RT_LLVM_CUDA} MODULE
        ATH_RT_LLVM # export name
        backend/llvm/runtime/runtime_export.h # export file name

        api.cpp
        CudaContext.cpp
        CudaDevice.cpp
        CudaEvent.cpp
  )

target_link_libraries(${POLAR_RT_LLVM_CUDA} PRIVATE CUDA::cuda_driver)
