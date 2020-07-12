configure_file(config.h.in ${CMAKE_CURRENT_BINARY_DIR}/config.h @ONLY)

add_polar_library(polar_generic_be_driver STATIC
  DynamicLibrary_posix.cpp
  RuntimeDriver.cpp)
add_library(PolarAI::generic_backend_driver ALIAS polar_generic_be_driver)
target_include_directories(polar_generic_be_driver PUBLIC 
                           $<BUILD_INTERFACE:${CMAKE_CURRENT_BINARY_DIR}>)

target_include_directories(polar_generic_be_driver PRIVATE 
                           ${CMAKE_CURRENT_BINARY_DIR})

if (UNIX)
  target_link_libraries(polar_generic_be_driver PRIVATE dl)
endif()

install(TARGETS polar_generic_be_driver EXPORT PolarAIConfig
        RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR}
        COMPONENT PolarAI_Runtime
        LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
        COMPONENT PolarAI_Runtime
        NAMELINK_COMPONENT PolarAI_Development
        ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR}
        COMPONENT PolarAI_Development
        )
