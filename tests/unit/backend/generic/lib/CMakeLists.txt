include_directories(${PROJECT_SOURCE_DIR}/src/backend/generic)

file(GLOB_RECURSE BACKEND_TEST_SRC "*.cpp")

add_polar_executable(TestUnitGenericBackendRunnable ${BACKEND_TEST_SRC})
target_link_libraries(TestUnitGenericBackendRunnable PUBLIC
        PolarAI::framework
        PolarAI::generic_backend
        PolarAI::utils
        Threads::Threads
        gtest gtest_main
        AthenaDep::googletest)

add_test(NAME GenericBackendUnitTest COMMAND TestUnitGenericBackendRunnable)
