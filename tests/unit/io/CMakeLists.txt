file(GLOB_RECURSE MODEL_TEST_SRC "*.cpp")

add_polar_executable(TestUnitModelRunnable ${MODEL_TEST_SRC})
target_link_libraries(TestUnitModelRunnable PUBLIC
        PolarAI::framework
        PolarAI::utils
        Threads::Threads
        AthenaDep::googletest)

add_test(NAME ModelUnitTest COMMAND TestUnitModelRunnable)
