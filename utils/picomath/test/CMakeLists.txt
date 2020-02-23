file(GLOB_RECURSE TEST_SRC "*.cpp")

add_executable(PicomathUnitTestOmp ${TEST_SRC})
target_link_libraries(PicomathUnitTestOmp PRIVATE AthenaDep::googletest picomath_omp)

add_test(Name PicomathUnitTest COMMAND PicomathUnitTest)