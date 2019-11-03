add_athena_integration_test(
        TARGET_NAME JIT
        SRCS vector_add.cpp gemm.cpp linreg.cpp
        LIBS athena
)

target_include_directories(TestIntegrationJITRunnable PUBLIC ${LLVM_INCLUDE_DIRS})
target_compile_definitions(TestIntegrationJITRunnable PUBLIC ${LLVM_DEFINITIONS})
