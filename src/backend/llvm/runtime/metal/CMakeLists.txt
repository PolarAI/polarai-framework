set(POLAR_RT_LLVM_METAL "PolarLLVMMetalRT" CACHE STRING "" FORCE)
set(POLAR_HAS_METAL "true" CACHE STRING "" FORCE)

add_definitions("-x objective-c++")
add_athena_library(${POLAR_RT_LLVM_METAL} SHARED
        ATH_RT_LLVM # export name
        backend/llvm/runtime/runtime_export.h # export file name

        api.mm
        MetalContext.mm
        MetalDevice.mm
        )

target_link_libraries(${POLAR_RT_LLVM_METAL} PRIVATE
        "-framework Metal"
        "-framework Cocoa"
        "-framework AppKit"
        "-framework CoreData"
        "-framework Foundation"
        )