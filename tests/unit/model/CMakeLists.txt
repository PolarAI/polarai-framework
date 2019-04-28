add_athena_executable(TestUnitModelRunnable NativeModel.cpp)
target_link_libraries(TestUnitModelRunnable PRIVATE
        athena
        Threads::Threads
        AthenaDep::googletest)

add_test(NAME ModelUnitTest COMMAND TestUnitModelRunnable)
