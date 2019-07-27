find_package(LLVM CONFIG)
llvm_map_components_to_libnames(llvm_libs core irreader support transformutils passes codegen orcjit x86asmparser x86codegen lto)

add_athena_integration_test(
        TARGET_NAME JIT
        SRCS vector_add.cpp
        LIBS athena ${llvm_libs} MemoryLoader
)

target_include_directories(TestIntegrationJITRunnable PUBLIC ${LLVM_INCLUDE_DIRS})
target_compile_definitions(TestIntegrationJITRunnable PUBLIC ${LLVM_DEFINITIONS})
