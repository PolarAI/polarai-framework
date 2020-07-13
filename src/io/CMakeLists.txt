add_polar_library(polar_io OBJECT 
        DotModel.cpp)
add_library(PolarAI::io ALIAS polar_io)
target_include_directories(polar_io PUBLIC 
                           $<BUILD_INTERFACE:${CMAKE_CURRENT_BINARY_DIR}>)

install(FILES ${CMAKE_CURRENT_BINARY_DIR}/polar_io_export.h
        DESTINATION ${CMAKE_INSTALL_INCLUDEDIR})
