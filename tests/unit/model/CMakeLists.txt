file(GLOB_RECURSE MODEL_TEST_SRC "*.cpp")

add_athena_executable(TestUnitModelRunnable ${MODEL_TEST_SRC})
target_link_libraries(TestUnitModelRunnable PUBLIC
        AthenaCore
        AthenaOperation
        AthenaLoaders
        AthenaUtils
        AthenaModel
        Threads::Threads
        AthenaDep::googletest)

add_test(NAME ModelUnitTest COMMAND TestUnitModelRunnable)
