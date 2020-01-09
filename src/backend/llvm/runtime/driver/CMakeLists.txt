set(ATH_RT_NEW_LLVM_DRIVER "AthenaLLVMRuntimeDriver" CACHE STRING "" FORCE)

set(OS_UTILS)

if (UNIX)
    set(OS_UTILS os_posix.cpp)
elseif (WIN32)
    message(FATAL_ERROR "Runtime Driver is not supported by your OS")
endif ()

add_athena_library(${ATH_RT_NEW_LLVM_DRIVER} STATIC
        ATH_RT_LLVM_DRIVER # export name
        backend/llvm/driver/driver_export.h # export file name
        RuntimeDriver.cpp ${OS_UTILS})
target_link_libraries(${ATH_RT_NEW_LLVM_DRIVER} PRIVATE ${ATH_CORE})

if (UNIX AND NOT APPLE)
    target_link_libraries(${ATH_RT_NEW_LLVM_DRIVER} PRIVATE dl.so)
endif ()
