configure_file(config.h.in ${CMAKE_CURRENT_BINARY_DIR}/config.h @ONLY)

add_polar_library(polar_generic_be_driver STATIC
  DynamicLibrary_posix.cpp
  RuntimeDriver.cpp)
add_library(PolarAI::generic_backend_driver ALIAS polar_generic_be_driver)

target_include_directories(PolarAI::generic_backend_driver PRIVATE 
                           ${CMAKE_CURRENT_BINARY_DIR})

if (UNIX)
  target_link_libraries(PolarAI::generic_backend_driver PRIVATE dl)
endif()

install(TARGETS PolarAI::generic_backend_driver EXPORT PolarAIConfig
        RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR}
        COMPONENT PolarAI_Runtime
        LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
        COMPONENT PolarAI_Runtime
        NAMELINK_COMPONENT PolarAI_Development
        ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR}
        COMPONENT PolarAI_Development
        )
