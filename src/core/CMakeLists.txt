file(GLOB_RECURSE CORE_SRC "*.cpp")

add_polar_library(polar_core OBJECT ${CORE_SRC})
add_library(PolarAI::core ALIAS polar_core)
target_include_directories(polar_core PUBLIC 
                           $<BUILD_INTERFACE:${CMAKE_CURRENT_BINARY_DIR}>)

install(FILES ${CMAKE_CURRENT_BINARY_DIR}/polar_core_export.h
        DESTINATION ${CMAKE_INSTALL_INCLUDEDIR})
