set(POLAR_RT_LLVM_METAL "PolarLLVMMetalRT" CACHE STRING "" FORCE)
set(POLAR_HAS_METAL "true" CACHE STRING "" FORCE)

set(SPIRV_CROSS_EXCEPTIONS_TO_ASSERTIONS ON CACHE BOOL "" FORCE)
add_subdirectory(${PROJECT_SOURCE_DIR}/contrib/SPIRV-Cross
                 ${PROJECT_BINARY_DIR}/contrib/SPIRV-Cross)

add_definitions("-x objective-c++")
add_polar_library(polar_rt_metal SHARED
        api.mm
        MetalContext.mm
        MetalDevice.mm
        MetalEvent.mm
        spirv_converter.cpp
        )
target_include_directories(polar_rt_metal PUBLIC 
                           $<BUILD_INTERFACE:${CMAKE_CURRENT_BINARY_DIR}>)

target_compile_definitions(polar_rt_metal PRIVATE SPIRV_CROSS_EXCEPTIONS_TO_ASSERTIONS=1)

target_link_libraries(polar_rt_metal PRIVATE
        "-framework Metal"
        "-framework Cocoa"
        "-framework AppKit"
        "-framework CoreData"
        "-framework Foundation"
        spirv-cross-msl
        spirv-cross-util
        spirv-cross-core
        PolarAI::utils
        )

set_property(GLOBAL APPEND PROPERTY POLAR_AVAILABLE_RUNTIMES polar_rt_metal)

install(TARGETS polar_rt_metal EXPORT PolarAIConfig
        RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR}
        COMPONENT PolarAI_Runtime
        LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
        COMPONENT PolarAI_Runtime
        NAMELINK_COMPONENT PolarAI_Development
        ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR}
        COMPONENT PolarAI_Development
        )
