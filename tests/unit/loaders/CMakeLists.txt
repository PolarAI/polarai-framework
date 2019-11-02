add_athena_executable(TestUnitLoadersRunnable MemoryLoader.cpp)
target_link_libraries(TestUnitLoadersRunnable PRIVATE
        athena
        Threads::Threads
        AthenaDep::googletest)

add_test(NAME LoadersUnitTest COMMAND TestUnitLoadersRunnable)
