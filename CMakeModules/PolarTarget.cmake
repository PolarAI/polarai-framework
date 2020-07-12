include(PolarRuntime)

function(add_polar_library target_name modifier)
    set(source_list ${ARGN})
    add_library(${target_name} ${modifier} ${source_list})

    if (NOT "${modifier}" STREQUAL "INTERFACE")
        polar_disable_rtti(${target_name})
        polar_disable_exceptions(${target_name})
    endif ()

    if (UNIX)
        target_compile_options(${target_name} PRIVATE -fPIC -Werror)
    endif ()

    include(GenerateExportHeader)
    generate_export_header(${target_name})

    set_target_properties(${target_name} PROPERTIES
                BUILD_RPATH "${CMAKE_BUILD_RPATH};${PROJECT_BINARY_DIR}/lib"
                LIBRARY_OUTPUT_DIRECTORY ${PROJECT_BINARY_DIR}/lib 
                RUNTIME_OUTPUT_DIRECTORY ${PROJECT_BINARY_DIR}/bin)

    find_package(codecov)
    add_coverage(${target_name})
endfunction()

function(add_polar_executable target_name)
    set(source_list ${ARGN})
    add_executable(${target_name} ${modifier} ${source_list})
    polar_disable_rtti(${target_name})
    polar_disable_exceptions(${target_name})
    set_target_properties(${target_name} PROPERTIES
                BUILD_RPATH "${CMAKE_BUILD_RPATH};${PROJECT_BINARY_DIR}/lib"
                RUNTIME_OUTPUT_DIRECTORY ${PROJECT_BINARY_DIR}/bin)
endfunction()
