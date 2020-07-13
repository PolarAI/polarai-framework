file(GLOB_RECURSE OPERATION_SRC "*.cpp")

add_polar_library(polar_operation OBJECT
        ${OPERATION_SRC})
add_library(PolarAI::operation ALIAS polar_operation)
target_include_directories(polar_operation PUBLIC 
                           $<BUILD_INTERFACE:${CMAKE_CURRENT_BINARY_DIR}>)

install(FILES ${CMAKE_CURRENT_BINARY_DIR}/polar_operation_export.h
        DESTINATION ${CMAKE_INSTALL_INCLUDEDIR})
