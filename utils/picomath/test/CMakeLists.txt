file(GLOB_RECURSE TEST_SRC "*.cpp")

find_package(Threads)

add_executable(PicomathUnitTestOmp ${TEST_SRC})
target_link_libraries(PicomathUnitTestOmp PRIVATE AthenaDep::googletest picomath_omp
        Threads::Threads)

add_test(NAME PicomathUnitTest COMMAND PicomathUnitTestOmp)
