option(ATHENA_BUILD_STATIC "OFF")

set(LIB_TYPE SHARED)
if (ATHENA_BUILD_STATIC)
    set(LIB_TYPE STATIC)
endif ()

file(WRITE ${CMAKE_CURRENT_BINARY_DIR}/null.cpp "")

add_athena_library(athena ${LIB_TYPE} ${CMAKE_CURRENT_BINARY_DIR}/null.cpp)

add_subdirectory(backend)
add_subdirectory(core)
add_subdirectory(ops)
add_subdirectory(loaders)
add_subdirectory(model)

# Define headers for this library. PUBLIC headers are used for
# compiling the library, and will be added to consumers' build
# paths.
target_include_directories(athena PUBLIC
        $<BUILD_INTERFACE:${CMAKE_SOURCE_DIR}/include>
        $<INSTALL_INTERFACE:include>)

set(WHOLE_ARCHIVE)
set(NOWHOLE_ARCHIVE)

if (APPLE)
    set(WHOLE_ARCHIVE -Wl,-all_load)
    set(NOWHOLE_ARCHIVE )
elseif (UNIX)
    set(WHOLE_ARCHIVE -Wl,--whole-archive)
    set(NOWHOLE_ARCHIVE -Wl,--no-whole-archive)
endif()

target_link_libraries(athena
        PRIVATE
        ${WHOLE_ARCHIVE}
        backend-llvm
        athena-core
        athena-ops
        athena-model
        ${NOWHOLE_ARCHIVE})

target_compile_features(athena
        PUBLIC
        cxx_inline_namespaces
        cxx_strong_enums
        cxx_variadic_templates
        PRIVATE
        cxx_override
        cxx_final
        cxx_attributes
        cxx_auto_type
        cxx_constexpr
        cxx_defaulted_functions
        cxx_defaulted_move_initializers)

install(TARGETS athena EXPORT AthenaConfig
        LIBRARY DESTINATION lib)

install(EXPORT AthenaConfig
        NAMESPACE Athena::
        DESTINATION share/athena/cmake)

# This makes the project importable from the build directory
export(TARGETS athena athena-core athena-ops FILE AthenaConfig.cmake)