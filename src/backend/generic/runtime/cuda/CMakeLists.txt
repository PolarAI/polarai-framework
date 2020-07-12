set(POLAR_HAS_CUDA "true" CACHE STRING "" FORCE)

add_polar_library(polar_rt_cuda MODULE
        api.cpp
        CudaContext.cpp
        CudaDevice.cpp
        CudaEvent.cpp
  )
target_include_directories(polar_rt_cuda PUBLIC 
                           $<BUILD_INTERFACE:${CMAKE_CURRENT_BINARY_DIR}>)

target_link_libraries(polar_rt_cuda PRIVATE CUDA::cuda_driver)
