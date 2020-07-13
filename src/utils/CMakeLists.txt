file(GLOB_RECURSE UTILS_SRC "*.cpp")

add_polar_library(polar_utils STATIC
        ${UTILS_SRC})
add_library(PolarAI::utils ALIAS polar_utils)
target_include_directories(polar_utils PUBLIC 
                           $<BUILD_INTERFACE:${CMAKE_CURRENT_BINARY_DIR}>)

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
