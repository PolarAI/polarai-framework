# This file contains mandatory dependencies

# Google test is the primary testing framework for the project
set(gtest_force_shared_crt ON CACHE BOOL "" FORCE)
AthenaAddDependency(
        TARGET_NAME googletest
        GIT_URL https://github.com/google/googletest.git
        GIT_TAG release-1.10.0
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
        GIT_URL https://github.com/alexbatashev/effcee.git
        GIT_TAG python3
        INCLUDE_PATH .
)
