set(ATH_RT_LLVM_SUPPORT "AthenaLLVMSupport" CACHE STRING "" FORCE)
set(ATH_RT_LLVM_SUPPORT_OBJ "AthenaLLVMSupportObj" CACHE STRING "" FORCE)
add_athena_library(${ATH_RT_LLVM_SUPPORT_OBJ} OBJECT 
        ATH_RT_SUPPORT # export name
        backend/llvm/runtime/support/export.h # export file name
        routines.cpp)

add_athena_library(${ATH_RT_LLVM_SUPPORT} STATIC 
        ATH_RT_SUPPORT # export name
        backend/llvm/runtime/support/export.h # export file name
        $<TARGET_OBJECTS:${ATH_RT_LLVM_SUPPORT_OBJ}>)

install(TARGETS ${ATH_RT_LLVM_SUPPORT} EXPORT AthenaConfig
        RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR}
        COMPONENT Athena_Runtime
        LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
        COMPONENT Athena_Runtime
        NAMELINK_COMPONENT Athena_Development
        ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR}
        COMPONENT Athena_Development
        )
