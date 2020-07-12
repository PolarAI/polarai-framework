file(GLOB_RECURSE UTILS_SRC "*.cpp")

add_polar_library(polar_utils STATIC
        ${UTILS_SRC})
add_library(PolarAI::utils ALIAS polar_utils)
target_include_directories(polar_utils PUBLIC 
                           $<BUILD_INTERFACE:${CMAKE_CURRENT_BINARY_DIR}>)

install(FILES ${PROJECT_BINARY_DIR}/polar_utils_export.h
        DESTINATION ${CMAKE_INSTALL_INCLUDEDIR})
