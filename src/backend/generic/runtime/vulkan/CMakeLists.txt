set(POLAR_HAS_VULKAN "true" CACHE STRING "" FORCE)

add_polar_library(polar_rt_vulkan SHARED 
        api.cpp
        VulkanContext.cpp
        VulkanDevice.cpp
        VulkanEvent.cpp
  )

target_link_libraries(polar_rt_vulkan PRIVATE Vulkan::Vulkan)
