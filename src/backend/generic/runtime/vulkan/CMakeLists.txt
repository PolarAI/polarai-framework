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

target_link_libraries(${POLAR_RT_LLVM_VULKAN} PRIVATE Vulkan::Vulkan)
