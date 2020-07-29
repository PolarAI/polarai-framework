option(POLARAI_USE_MIMALLOC "" OFF)

if (POLARAI_USE_MIMALLOC)
  set(MI_OVERRIDE OFF CACHE BOOL "" FORCE)

  FetchContent_Declare(mimalloc
    GIT_REPOSITORY https://github.com/microsoft/mimalloc.git
    GIT_TAG v1.6.3
  )

  FetchContent_MakeAvailable(mimalloc)
  add_definitions(-DHAS_MIMALLOC)
  set(HAS_CUSTOM_ALLOCATOR TRUE)
endif()

if (NOT HAS_CUSTOM_ALLOCATOR)
  message(WARNING "Custom allocators are disabled. This may have performance impacts.")
endif()

file(GLOB_RECURSE UTILS_SRC "*.cpp")

add_polar_library(polar_utils SHARED 
        ${UTILS_SRC})
add_library(PolarAI::utils ALIAS polar_utils)
target_include_directories(polar_utils PUBLIC 
                           $<BUILD_INTERFACE:${CMAKE_CURRENT_BINARY_DIR}>)
if (POLARAI_USE_MIMALLOC)
  target_link_libraries(polar_utils PRIVATE mimalloc-static)
endif()

install(FILES ${CMAKE_CURRENT_BINARY_DIR}/polar_utils_export.h
        DESTINATION ${CMAKE_INSTALL_INCLUDEDIR})
install(TARGETS polar_utils EXPORT PolarAIConfig
        RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR}
        COMPONENT PolarAI_Runtime
        LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
        COMPONENT PolarAI_Runtime
        NAMELINK_COMPONENT PolarAI_Development
        ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR}
        COMPONENT PolarAI_Development
        )
export(TARGETS polar_utils FILE PolarAIConfig.cmake)
