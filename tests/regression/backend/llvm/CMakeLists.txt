file(GLOB_RECURSE REG_TEST_SRC "*.cpp")

include(AthenaIntegrationTest)

add_athena_integration_test(
        TARGET_NAME LLVMRegression
        SRCS ${REG_TEST_SRC}
        LIBS athena AthenaDep::effcee
)
