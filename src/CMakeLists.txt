include(CMakePackageConfigHelpers)
include(GNUInstallDirs)

set(CMAKE_CXX_VISIBILITY_PRESET hidden)
set(CMAKE_VISIBILITY_INLINES_HIDDEN 1)

# --- Global Options ---
option(ATHENA_BUILD_STATIC "OFF")

set(LIB_TYPE SHARED)
if (ATHENA_BUILD_STATIC)
    set(LIB_TYPE STATIC)
endif ()

if (NOT APPLE)
    set(CMAKE_INSTALL_RPATH $ORIGIN)
endif ()

# This dummy file is needed to construct archive library
file(WRITE ${CMAKE_CURRENT_BINARY_DIR}/null.cpp "")

# --- Library targets ---
add_athena_library(athena ${LIB_TYPE}
        ATHENA # export name
        athena_export.h
        ${CMAKE_CURRENT_BINARY_DIR}/null.cpp)

add_subdirectory(backend)
add_subdirectory(core)
add_subdirectory(ops)
add_subdirectory(loaders)
add_subdirectory(model)

# --- Library options ---
set_target_properties(athena PROPERTIES
        SOVERSION ${ATHENA_MAJOR_VERSION}
        VERSION ${ATHENA_VERSION})

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
    set(NOWHOLE_ARCHIVE)
elseif (UNIX)
    set(WHOLE_ARCHIVE -Wl,--whole-archive)
    set(NOWHOLE_ARCHIVE -Wl,--no-whole-archive)
endif ()

target_link_libraries(athena
        PRIVATE
        ${WHOLE_ARCHIVE}
        ${ATH_BACKEND_LLVM}
        ${ATH_CORE}
        ${ATH_MEM_LOADER}
        ${ATH_OPS}
        ${ATH_MODEL}
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

# --- Library installation ---
install(TARGETS athena EXPORT AthenaConfig
        RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR}
        COMPONENT Athena_Runtime
        LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
        COMPONENT Athena_Runtime
        NAMELINK_COMPONENT Athena_Development
        ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR}
        COMPONENT Athena_Development
        )

install(EXPORT AthenaConfig
        NAMESPACE Athena::
        DESTINATION share/athena/cmake)

write_basic_package_version_file(
        AthenaConfigVersion.cmake
        VERSION ${ATHENA_VERSION}
        COMPATIBILITY SameMajorVersion
)

# This makes the project importable from the build directory
export(TARGETS athena ${ATH_CORE} FILE AthenaConfig.cmake)