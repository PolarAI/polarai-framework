cmake_minimum_required(VERSION 3.13)
project(athena NONE)

set(CMAKE_MODULE_PATH ${PROJECT_SOURCE_DIR}/CMakeModules)

set(ATHENA_MAJOR_VERSION "0")
set(ATHENA_MINOR_VERSION "1")
set(ATHENA_PATCH_VERSION "0")
set(ATHENA_VERSION "${ATHENA_MAJOR_VERSION}.${ATHENA_MINOR_VERSION}.${ATHENA_PATCH_VERSION}")

option(ATHENA_DISABLE_TESTS "OFF")
option(ATHENA_DOCS_ONLY "OFF")
set(ATHENA_USE_SANITIZERS "OFF" CACHE STRING "Specify Google Sanitizers")

# choose new behaviour for CMP0042
# see http://www.cmake.org/cmake/help/v3.0/policy/CMP0042.html for more details
if (POLICY CMP0042)
    cmake_policy(SET CMP0042 NEW)
endif (POLICY CMP0042)

include(AthenaDocs)
enable_docs()
set(CMAKE_CXX_STANDARD 17)
list(APPEND CMAKE_MODULE_PATH ${PROJECT_SOURCE_DIR}/CMakeModules)

if (NOT ATHENA_DOCS_ONLY)

    enable_language(C)
    enable_language(CXX)

    set(CMAKE_CXX_STANDARD 17)

    # todo remove hacks http://lists.llvm.org/pipermail/llvm-dev/2016-May/099874.html
    if (UNIX AND NOT APPLE)
        if ("${CMAKE_CXX_COMPILER_ID}" STREQUAL "Clang")
            add_link_options("-rdynamic")
        else ()
            add_link_options("-export-dynamic")
        endif ()
    endif ()

    include(AthenaTarget)
    include(AthenaEnableSanitizers)
    include(AthenaDependency)
    include(dependencies.cmake)

    enable_testing()
    enable_sanitizers()

    include_directories(${CMAKE_SOURCE_DIR}/include)
    install(DIRECTORY ${CMAKE_SOURCE_DIR}/include/athena DESTINATION include)
    install(DIRECTORY ${CMAKE_BINARY_DIR}/export/athena DESTINATION include)

    add_definitions(-DATHENA_MAJOR_VERSION="${ATHENA_MAJOR_VERSION}")
    add_definitions(-DATHENA_MINOR_VERSION="${ATHENA_MINOR_VERSION}")
    if (DEFINED $ENV{ATHENA_CI_BUILD_NUMBER})
        add_definitions(-DATHENA_PATCH_VERSION="$ENV{ATHENA_CI_BUILD_NUMBER}")
    else ()
        add_definitions(-DATHENA_PATCH_VERSION="${ATHENA_PATCH_VERSION}")
    endif ()

    if ("${CMAKE_BUILD_TYPE}" STREQUAL "Debug")
        message(STATUS "Debug build contains assertions")
        add_definitions(-DDEBUG)
    endif ()

    add_subdirectory(utils/athena-tablegen)
    add_subdirectory(utils/test_runner)
    add_subdirectory(src)

    if (NOT ATHENA_DISABLE_TESTS)
        add_subdirectory(tests)
        coverage_evaluate()
    endif ()

    set(CPACK_PACKAGE_VENDOR "Athena")
    set(CPACK_PACKAGE_DESCRIPTION_SUMMARY "High performance deep learning library")
    set(CPACK_PACKAGE_VERSION_MAJOR ${ATHENA_MAJOR_VERSION})
    set(CPACK_PACKAGE_VERSION_MINOR ${ATHENA_MINOR_VERSION})
    set(CPACK_PACKAGE_VERSION_PATCH ${ATHENA_PATCH_VERSION})
    set(CPACK_RESOURCE_FILE_LICENSE "${CMAKE_CURRENT_SOURCE_DIR}/LICENSE.md")
    set(CPACK_RESOURCE_FILE_README "${CMAKE_CURRENT_SOURCE_DIR}/README.md")
    include(CPack)
endif ()
