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
    elseif(WIN32)
        target_compile_definitions(${target_name} PUBLIC /D_ITERATOR_DEBUG_LEVEL=0 /D_HAS_ITERATOR_DEBUGGING=0)
        target_compile_options(${target_name} PRIVATE /Zp8)
    endif ()

    include(GenerateExportHeader)
    generate_export_header(${target_name})

    set_target_properties(${target_name} PROPERTIES
                BUILD_RPATH "${CMAKE_BUILD_RPATH};${PROJECT_BINARY_DIR}/lib"
                LIBRARY_OUTPUT_DIRECTORY ${PROJECT_BINARY_DIR}/lib 
                RUNTIME_OUTPUT_DIRECTORY ${PROJECT_BINARY_DIR}/bin)
    
    # TODO this is a workaround for windows, which has concept of 
    # imported symbols. It must be removed in future.
    if (${modifier} MATCHES "OBJECT" AND NOT "${target_name}" MATCHES "polar_rt_support")
        target_compile_definitions(${target_name} PRIVATE 
                            polar_core_EXPORTS 
                            polar_operation_EXPORTS 
                            polar_io_EXPORTS
                            polar_utils_EXPORTS
                            polar_loaders_EXPORTS)
    endif()

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

    if(WIN32)
        target_compile_definitions(${target_name} PRIVATE -D_ITERATOR_DEBUG_LEVEL=0)
        target_compile_options(${target_name} PRIVATE /Zp8)
    endif ()
endfunction()
