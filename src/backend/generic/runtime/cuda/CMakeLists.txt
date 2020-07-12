set(POLAR_HAS_CUDA "true" CACHE STRING "" FORCE)

add_polar_library(polar_rt_cuda SHARED
        api.cpp
        CudaContext.cpp
        CudaDevice.cpp
        CudaEvent.cpp
  )
target_include_directories(polar_rt_cuda PUBLIC 
                           $<BUILD_INTERFACE:${CMAKE_CURRENT_BINARY_DIR}>)

target_link_libraries(polar_rt_cuda PRIVATE CUDA::cuda_driver PolarAI::utils)

set_property(GLOBAL APPEND PROPERTY POLAR_AVAILABLE_RUNTIMES polar_rt_cuda)

install(TARGETS polar_rt_cuda EXPORT PolarAIConfig
        RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR}
        COMPONENT PolarAI_Runtime
        LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
        COMPONENT PolarAI_Runtime
        NAMELINK_COMPONENT PolarAI_Development
        ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR}
        COMPONENT PolarAI_Development
        )
