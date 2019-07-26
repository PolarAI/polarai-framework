file(GLOB_RECURSE CORE_TEST_SRC "*.cpp")

add_athena_executable(TestUnitCoreRunnable ${CORE_TEST_SRC})
target_link_libraries(TestUnitCoreRunnable PRIVATE
        athena-core
        Threads::Threads
        AthenaDep::googletest)

add_test(NAME CoreUnitTest COMMAND TestUnitCoreRunnable)
