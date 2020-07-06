function(athena_disable_rtti target_name)
    if (WIN32)
        set(NEW_COMPILE_OPTIONS "/GR-")
    else()
        set(NEW_COMPILE_OPTIONS "-fno-rtti")
    endif()

    target_compile_options(${target_name} PRIVATE "${NEW_COMPILE_OPTIONS}")
endfunction(athena_disable_rtti)

function(athena_disable_exceptions target_name)
    if (WIN32)
        set(NEW_COMPILE_OPTIONS "/EHsc-")
    else()
        set(NEW_COMPILE_OPTIONS "-fno-exceptions")
    endif()

    target_compile_options(${target_name} PRIVATE "${NEW_COMPILE_OPTIONS}")
endfunction(athena_disable_exceptions)
