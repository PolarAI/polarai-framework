# This file contains mandatory dependencies

# yaml-cpp is used to parse integration test config files
set(YAML_CPP_BUILD_TESTS OFF CACHE BOOL "" FORCE)
set(YAML_CPP_BUILD_CONTRIB OFF CACHE BOOL "" FORCE)
set(YAML_CPP_BUILD_TOOLS OFF CACHE BOOL "" FORCE)
AthenaAddDependency(
        TARGET_NAME yaml-cpp
        GIT_URL https://github.com/jbeder/yaml-cpp.git
        GIT_TAG yaml-cpp-0.6.2
)

# Boost filesystem is used to process paths in integration tests
AthenaAddDependency(
        TARGET_NAME Boost
        PACKAGE Boost
        COMPONENTS filesystem
)

# Google test is the primary testing framework for the project
AthenaAddDependency(
        TARGET_NAME googletest
        GIT_URL https://github.com/google/googletest.git
        GIT_TAG release-1.8.1
        PACKAGE GTest
        LIBRARIES gtest gtest_main
        INCLUDE_PATH googletest/include
)

set(RE2_BUILD_TESTING OFF CACHE BOOL "" FORCE)
AthenaAddDependency(
        TARGET_NAME re2
        GIT_URL https://github.com/google/re2.git
        GIT_TAG master
        INCLUDE_PATH .
)

set(EFFCEE_BUILD_TESTING OFF CACHE BOOL "" FORCE)
set(EFFCEE_BUILD_SAMPLES OFF CACHE BOOL "" FORCE)
AthenaAddDependency(
        TARGET_NAME effcee
        GIT_URL https://github.com/google/effcee.git
        GIT_TAG master
        INCLUDE_PATH .
)