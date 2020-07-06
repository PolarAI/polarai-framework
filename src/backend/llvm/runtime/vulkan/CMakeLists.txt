set(POLAR_RT_LLVM_VULKAN "PolarLLVMVulkanRT" CACHE STRING "" FORCE)
set(POLAR_HAS_VULKAN "true" CACHE STRING "" FORCE)

add_athena_library(${POLAR_RT_LLVM_VULKAN} SHARED 
        ATH_RT_LLVM # export name
        backend/llvm/runtime/runtime_export.h # export file name

        api.cpp
        VulkanContext.cpp
        VulkanDevice.cpp
        VulkanEvent.cpp
  )

set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -fPIC")
add_subdirectory(${PROJECT_SOURCE_DIR}/contrib/volk 
                 ${PROJECT_BINARY_DIR}/contrib/volk)
target_link_libraries(${POLAR_RT_LLVM_VULKAN} PRIVATE volk)
target_compile_definitions(${POLAR_RT_LLVM_VULKAN} PRIVATE VK_NO_PROTOTYPES)
