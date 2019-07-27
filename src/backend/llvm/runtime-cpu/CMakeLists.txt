find_package(BLAS)

add_athena_library(runtime-llvm-cpu SHARED
        add.cpp
        allocate.cpp
        fill.cpp
        ${WRAPPER_CPP})
add_dependencies(runtime-llvm-cpu wrapper_cpp)

if (APPLE)
    set_property(TARGET runtime-llvm-cpu APPEND_STRING PROPERTY LINK_FLAGS " -undefined dynamic_lookup")
elseif (UNIX)
    target_link_options(runtime-llvm-cpu PUBLIC "-Wl,--allow-shlib-undefined")
endif ()

install(TARGETS runtime-llvm-cpu EXPORT AthenaConfig
        LIBRARY DESTINATION lib/runtime)
