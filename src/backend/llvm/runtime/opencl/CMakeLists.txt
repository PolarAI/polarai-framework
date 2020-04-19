set(ATH_RT_LLVM_OCL "AthenaLLVMOCLRT" CACHE STRING "" FORCE)
add_athena_library(${ATH_RT_LLVM_OCL} MODULE
        ATH_RT_LLVM # export name
        backend/llvm/runtime/runtime_export.h # export file name
        api.cpp
        OpenClDevice.cpp
        BufferAllocator.cpp)
target_compile_definitions(${ATH_RT_LLVM_OCL} PRIVATE CL_TARGET_OPENCL_VERSION=220)

target_link_libraries(${ATH_RT_LLVM_OCL} PRIVATE OpenCL::OpenCL)
