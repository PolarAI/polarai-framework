set(POLAR_RT_LLVM_METAL "PolarLLVMMetalRT" CACHE STRING "" FORCE)
set(POLAR_HAS_METAL "true" CACHE STRING "" FORCE)

set(SPIRV_CROSS_EXCEPTIONS_TO_ASSERTIONS ON CACHE BOOL "" FORCE)
add_subdirectory(${PROJECT_SOURCE_DIR}/contrib/SPIRV-Cross
                 ${PROJECT_BINARY_DIR}/contrib/SPIRV-Cross)

add_definitions("-x objective-c++")
add_athena_library(${POLAR_RT_LLVM_METAL} SHARED
        ATH_RT_LLVM # export name
        backend/llvm/runtime/runtime_export.h # export file name

        api.mm
        MetalContext.mm
        MetalDevice.mm
        MetalEvent.mm
        spirv_converter.cpp
        )

target_compile_definitions(${POLAR_RT_LLVM_METAL} PRIVATE SPIRV_CROSS_EXCEPTIONS_TO_ASSERTIONS=1)

target_link_libraries(${POLAR_RT_LLVM_METAL} PRIVATE
        "-framework Metal"
        "-framework Cocoa"
        "-framework AppKit"
        "-framework CoreData"
        "-framework Foundation"
        spirv-cross-msl
        spirv-cross-util
        spirv-cross-core
        ${ATH_UTILS}
        )