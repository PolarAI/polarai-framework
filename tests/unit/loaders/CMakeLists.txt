add_athena_executable(TestUnitLoadersRunnable MemoryLoader.cpp)
target_link_libraries(TestUnitLoadersRunnable PRIVATE
        athena-core
        backend-llvm
        MemoryLoader
        Threads::Threads
        AthenaDep::googletest)

add_test(NAME LoadersUnitTest COMMAND TestUnitLoadersRunnable)
