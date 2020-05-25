cmake_minimum_required(VERSION 3.13)
project(athena-debugger CXX)

add_subdirectory(lib)

find_package(OpenGL)

if (OPENGL_FOUND) # fixme remove when ready for production
    add_subdirectory(src)
endif ()
