set(POLAR_HAS_VULKAN "true" CACHE STRING "" FORCE)

add_polar_library(polar_rt_vulkan SHARED 
        api.cpp
        VulkanContext.cpp
        VulkanDevice.cpp
        VulkanEvent.cpp
  )
target_include_directories(polar_rt_vulkan PUBLIC 
                           $<BUILD_INTERFACE:${CMAKE_CURRENT_BINARY_DIR}>)

target_link_libraries(polar_rt_vulkan PRIVATE Vulkan::Vulkan PolarAI::utils)

set_property(GLOBAL APPEND PROPERTY POLAR_AVAILABLE_RUNTIMES polar_rt_vulkan)

install(TARGETS polar_rt_vulkan EXPORT PolarAIConfig
        RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR}
        COMPONENT PolarAI_Runtime
        LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
        COMPONENT PolarAI_Runtime
        NAMELINK_COMPONENT PolarAI_Development
        ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR}
        COMPONENT PolarAI_Development
        )
