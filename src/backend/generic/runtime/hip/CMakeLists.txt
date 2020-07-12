set(POLAR_HAS_HIP "true" CACHE STRING "" FORCE)

find_library(ROCM_RUNTIME_LIBRARY hip_hcc
               PATHS "${HIP_PATH}/lib")
if (NOT ROCM_RUNTIME_LIBRARY)
  message(SEND_ERROR "Could not locate ROCm HIP runtime library")
else()
  message(STATUS "ROCm HIP runtime lib: ${ROCM_RUNTIME_LIBRARY}")
endif()

add_polar_library(polar_rt_hip MODULE
        api.cpp
        HIPContext.cpp
        HIPDevice.cpp
  )
target_include_directories(polar_rt_hip PUBLIC 
                           $<BUILD_INTERFACE:${CMAKE_CURRENT_BINARY_DIR}>)

target_link_libraries(polar_rt_hip PRIVATE ${ROCM_RUNTIME_LIBRARY})
target_compile_definitions(polar_rt_hip PRIVATE __HIP_PLATFORM_HCC__)
target_include_directories(polar_rt_hip SYSTEM PRIVATE
  "${HIP_PATH}/../include"
  "${HIP_PATH}/include")
