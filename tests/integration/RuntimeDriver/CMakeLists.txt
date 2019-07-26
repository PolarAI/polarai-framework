find_package(LLVM CONFIG)
llvm_map_components_to_libnames(llvm_libs core irreader support transformutils passes codegen orcjit x86asmparser x86codegen lto)

add_athena_integration_test(TARGET_NAME RuntimeDriver
        SRCS main.cpp
        LIBS runtime-driver)

target_include_directories(TestIntegrationRuntimeDriverRunnable PUBLIC ${LLVM_INCLUDE_DIRS})
target_compile_definitions(TestIntegrationRuntimeDriverRunnable PUBLIC ${LLVM_DEFINITIONS})
