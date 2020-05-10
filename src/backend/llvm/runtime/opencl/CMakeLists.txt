set(ATH_RT_LLVM_OCL "AthenaLLVMOCLRT" CACHE STRING "" FORCE)
add_athena_library(${ATH_RT_LLVM_OCL} MODULE
        ATH_RT_LLVM # export name
        backend/llvm/runtime/runtime_export.h # export file name
        api.cpp
        OpenClDevice.cpp
        BufferAllocator.cpp)
target_compile_definitions(${ATH_RT_LLVM_OCL} PRIVATE CL_TARGET_OPENCL_VERSION=220)

target_link_libraries(${ATH_RT_LLVM_OCL} PRIVATE OpenCL::OpenCL)

install(TARGETS ${ATH_RT_LLVM_OCL} EXPORT AthenaConfig
        RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR}
        COMPONENT Athena_Runtime
        LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
        COMPONENT Athena_Runtime
        NAMELINK_COMPONENT Athena_Development
        ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR}
        COMPONENT Athena_Development
        )
