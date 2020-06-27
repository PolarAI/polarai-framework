set(ATH_RT_LLVM_SYCL "AthenaLLVMSYCLRT" CACHE STRING "" FORCE)

include(AthenaSYCL)

add_athena_sycl_library(${ATH_RT_LLVM_SYCL} MODULE
        ATH_RT_LLVM # export name
        backend/llvm/runtime/runtime_export.h # export file name

        api.cpp
        BufferAllocator.cpp
        SYCLDevice.cpp
        SYCLEvent.cpp
        SYCLContext.cpp
  )
