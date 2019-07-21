include(AthenaRuntime)

function(add_athena_library target_name modifier)
    set(source_list ${ARGN})
    add_library(${target_name} ${modifier} ${source_list})

    if (NOT "${modifier}" STREQUAL "INTERFACE")
        athena_disable_rtti(${target_name})
        athena_disable_exceptions(${target_name})
    endif ()

    if (UNIX)
        target_compile_options(${target_name} PRIVATE "-fPIC")
    endif ()

    find_package(codecov)
    add_coverage(${target_name})
endfunction(add_athena_library)

function(add_athena_executable target_name)
    set(source_list ${ARGN})
    add_executable(${target_name} ${modifier} ${source_list})
    athena_disable_rtti(${target_name})
    athena_disable_exceptions(${target_name})
endfunction()

function(athena_add_linker_options target_name options)
    set_property(TARGET ${target_name} APPEND_STRING PROPERTY LINK_FLAGS " ${options}")
endfunction()
