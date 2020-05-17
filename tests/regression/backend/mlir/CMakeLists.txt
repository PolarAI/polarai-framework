file(GLOB_RECURSE REG_TEST_SRC "*.cpp")

add_athena_executable(
        TestRegressionMLIRRunnable
        ${REG_TEST_SRC})

target_link_libraries(TestRegressionMLIRRunnable PRIVATE
        athena
        AthenaDep::googletest
        AthenaDep::effcee)

add_test(NAME MLIRRegressionTest COMMAND TestRegressionMLIRRunnable)
