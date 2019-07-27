# This file contains mandatory dependencies

# yaml-cpp is used to parse integration test config files
set(YAML_CPP_BUILD_TESTS OFF)
set(YAML_CPP_BUILD_CONTRIB OFF)
set(YAML_CPP_BUILD_TOOLS OFF)
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
