file(GLOB_RECURSE LOADERS_SRC "*.cpp")

add_polar_library(polar_loaders OBJECT
        ${LOADERS_SRC})
add_library(PolarAI::loaders ALIAS polar_loaders)
target_include_directories(polar_loaders PUBLIC 
                           $<BUILD_INTERFACE:${CMAKE_CURRENT_BINARY_DIR}>)

install(FILES ${CMAKE_CURRENT_BINARY_DIR}/polar_loaders_export.h
        DESTINATION ${CMAKE_INSTALL_INCLUDEDIR})
