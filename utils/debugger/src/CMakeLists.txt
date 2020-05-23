find_package(OpenGL)

set(CONTRIB_DIR ${CMAKE_CURRENT_SOURCE_DIR}/../../../contrib/)

if (OPENGL_FOUND)
    set(GLFW_BUILD_DOCS OFF CACHE BOOL "" FORCE)
    add_subdirectory(${CONTRIB_DIR}/glfw ${CMAKE_CURRENT_BINARY_DIR}/glfw)
    set(SRC
            ${CONTRIB_DIR}/imgui/examples/imgui_impl_opengl3.cpp
            ${CONTRIB_DIR}/imgui/examples/imgui_impl_glfw.cpp
            ${CONTRIB_DIR}/imgui/imgui.cpp
            ${CONTRIB_DIR}/imgui/imgui_widgets.cpp
            ${CONTRIB_DIR}/imgui/imgui_draw.cpp
            ${CONTRIB_DIR}/glad/src/glad.c
            opengl/main.cpp
            )
    include_directories(${CONTRIB_DIR}/glad/include)
    include_directories(${CONTRIB_DIR}/glfw/include)

    set(TARGET_LIBS OpenGL::GL glfw)
endif ()

include_directories(${CONTRIB_DIR}/imgui)
include_directories(${CONTRIB_DIR}/imgui/examples)
include_directories(${CONTRIB_DIR}/imgui-flame-graph)
include_directories(${CONTRIB_DIR}/imnodes)

add_athena_executable(nndbg
        debugger.cpp
        widgets/graph.cpp
        ${CONTRIB_DIR}/imgui-flame-graph/imgui_widget_flamegraph.cpp
        ${CONTRIB_DIR}/imnodes/imnodes.cpp
        ${CONTRIB_DIR}/imgui/imgui.cpp
        ${CONTRIB_DIR}/imgui/imgui_widgets.cpp
        ${SRC}
        )
target_link_libraries(nndbg PRIVATE ${TARGET_LIBS})
