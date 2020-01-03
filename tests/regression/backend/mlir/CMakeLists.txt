file(GLOB_RECURSE REG_TEST_SRC "*.cpp")

include(AthenaIntegrationTest)

find_package(MLIR COMPONENTS IR)

add_athena_integration_test(
        TARGET_NAME MLIRRegression
        SRCS ${REG_TEST_SRC}
        LIBS athena AthenaDep::effcee ${MLIR_LIBS} ${ATH_BACKEND_LLVM_MLIR}
        /opt/llvm10/lib/libMLIRDialect.a
        /opt/llvm10/lib/libMLIREDSC.a
        /opt/llvm10/lib/libMLIRSupport.a
)
target_include_directories(TestIntegrationMLIRRegressionRunnable PRIVATE ${MLIR_INCLUDE_DIRS})
