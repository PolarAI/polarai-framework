cmake_minimum_required(VERSION 3.13)
project(polarai CXX)

set(CMAKE_MODULE_PATH ${PROJECT_SOURCE_DIR}/CMakeModules)

set(POLAR_MAJOR_VERSION "0")
set(POLAR_MINOR_VERSION "1")
set(POLAR_PATCH_VERSION "0")
set(POLAR_VERSION "${POLAR_MAJOR_VERSION}.${POLAR_MINOR_VERSION}.${POLAR_PATCH_VERSION}")

option(POLAR_DISABLE_TESTS "OFF")
set(POLAR_USE_SANITIZERS "OFF" CACHE STRING "Specify Google Sanitizers")

define_property(GLOBAL PROPERTY POLAR_AVAILABLE_RUNTIMES
    BRIEF_DOCS "A list of enabled generic runtimes"
    FULL_DOCS "Depending on system configuration a different set of runtimes may be supported. This property contains a list of enabled runtime targets")

# choose new behaviour for CMP0042
# see http://www.cmake.org/cmake/help/v3.0/policy/CMP0042.html for more details
if (POLICY CMP0042)
    cmake_policy(SET CMP0042 NEW)
endif (POLICY CMP0042)

set(CMAKE_CXX_STANDARD 17)
list(APPEND CMAKE_MODULE_PATH ${PROJECT_SOURCE_DIR}/CMakeModules)

# todo remove hacks http://lists.llvm.org/pipermail/llvm-dev/2016-May/099874.html
if (UNIX AND NOT APPLE)
    if ("${CMAKE_CXX_COMPILER_ID}" STREQUAL "Clang")
        add_link_options("-rdynamic")
    else ()
        add_link_options("-export-dynamic")
    endif ()
endif ()

include(PolarTarget)
include(AthenaEnableSanitizers)
include(AthenaDependency)
include(dependencies.cmake)

enable_testing()
enable_sanitizers()

include_directories(${CMAKE_SOURCE_DIR}/include)
install(DIRECTORY ${CMAKE_SOURCE_DIR}/include/polarai DESTINATION include)

add_definitions(-DATHENA_MAJOR_VERSION="${POLAR_MAJOR_VERSION}")
add_definitions(-DATHENA_MINOR_VERSION="${POLAR_MINOR_VERSION}")
if (DEFINED $ENV{ATHENA_CI_BUILD_NUMBER})
    add_definitions(-DATHENA_PATCH_VERSION="$ENV{ATHENA_CI_BUILD_NUMBER}")
else ()
    add_definitions(-DATHENA_PATCH_VERSION="${POLAR_PATCH_VERSION}")
endif ()

if ("${CMAKE_BUILD_TYPE}" STREQUAL "Debug")
    message(STATUS "Debug build contains assertions")
    add_definitions(-DDEBUG)
endif ()

if (NOT POLAR_NO_LLVM)
  add_subdirectory(utils/dialects)
  add_subdirectory(utils/PolarScript)
endif()
add_subdirectory(src)
add_subdirectory(utils/lspolar)

if (NOT POLAR_DISABLE_TESTS)
    add_subdirectory(tests)
    coverage_evaluate()
endif ()

set(CPACK_PACKAGE_VENDOR "PolarAI")
set(CPACK_PACKAGE_DESCRIPTION_SUMMARY "High performance deep learning library")
set(CPACK_PACKAGE_VERSION_MAJOR ${POLAR_MAJOR_VERSION})
set(CPACK_PACKAGE_VERSION_MINOR ${POLAR_MINOR_VERSION})
set(CPACK_PACKAGE_VERSION_PATCH ${POLAR_PATCH_VERSION})
set(CPACK_RESOURCE_FILE_LICENSE "${CMAKE_CURRENT_SOURCE_DIR}/LICENSE.md")
set(CPACK_RESOURCE_FILE_README "${CMAKE_CURRENT_SOURCE_DIR}/README.md")
include(CPack)
