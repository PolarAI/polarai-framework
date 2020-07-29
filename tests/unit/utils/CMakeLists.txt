file(GLOB_RECURSE UTILS_TEST_SRC "*.cpp")

add_polar_executable(TestUnitUtilsRunnable ${UTILS_TEST_SRC})
target_link_libraries(TestUnitUtilsRunnable PUBLIC
        PolarAI::utils
        Threads::Threads
        AthenaDep::googletest)

add_test(NAME UtilsUnitTest COMMAND TestUnitUtilsRunnable)

