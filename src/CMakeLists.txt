include(CMakePackageConfigHelpers)
include(GNUInstallDirs)

set(CMAKE_CXX_VISIBILITY_PRESET hidden)
set(CMAKE_VISIBILITY_INLINES_HIDDEN 1)

# --- Global Options ---
if (NOT APPLE)
    set(CMAKE_INSTALL_RPATH $ORIGIN)
endif ()

# --- Library targets ---
include_directories(${PROJECT_BINARY_DIR}/src/core)
include_directories(${PROJECT_BINARY_DIR}/src/io)
include_directories(${PROJECT_BINARY_DIR}/src/loaders)
include_directories(${PROJECT_BINARY_DIR}/src/operation)
include_directories(${PROJECT_BINARY_DIR}/src/utils)
add_subdirectory(core)
add_subdirectory(operation)
add_subdirectory(loaders)
add_subdirectory(io)
add_subdirectory(utils)

# Backends should go after core stuff
add_subdirectory(backend)

add_polar_library(polarai SHARED
        $<TARGET_OBJECTS:PolarAI::core>
        $<TARGET_OBJECTS:PolarAI::io>
        $<TARGET_OBJECTS:PolarAI::loaders>
        $<TARGET_OBJECTS:PolarAI::operation>
        )
add_library(PolarAI::framework ALIAS polarai)

target_link_libraries(polarai PRIVATE PolarAI::core PolarAI::io PolarAI::loaders
                              PolarAI::operation PolarAI::utils)

# --- Library options ---
set_target_properties(polarai PROPERTIES
        SOVERSION ${POLAR_MAJOR_VERSION}
        VERSION ${POLAR_VERSION})

# Define headers for this library. PUBLIC headers are used for
# compiling the library, and will be added to consumers' build
# paths.
target_include_directories(polarai PUBLIC
        $<BUILD_INTERFACE:${CMAKE_SOURCE_DIR}/include>
        $<INSTALL_INTERFACE:include>)

target_compile_features(polarai
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
install(TARGETS polarai EXPORT PolarAIConfig
        RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR}
        COMPONENT PolarAI_Runtime
        LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
        COMPONENT PolarAI_Runtime
        NAMELINK_COMPONENT PolarAI_Development
        ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR}
        COMPONENT PolarAI_Development
        )

install(EXPORT PolarAIConfig
        NAMESPACE PolarAI::
        DESTINATION share/polarai/cmake)

write_basic_package_version_file(
        PolarAIConfigVersion.cmake
        VERSION ${POLAR_VERSION}
        COMPATIBILITY SameMajorVersion
)

# This makes the project importable from the build directory
export(TARGETS polarai FILE PolarAIConfig.cmake)
