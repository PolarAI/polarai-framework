find_package(LLVM CONFIG)
llvm_map_components_to_libnames(llvm_libs core irreader support transformutils passes codegen orcjit x86asmparser x86codegen lto)

add_athena_integration_test(TARGET_NAME LegacyRuntimeDriver
        SRCS main.cpp
        LIBS ${ATH_RT_LLVM_DRIVER})

target_include_directories(TestIntegrationLegacyRuntimeDriverRunnable PUBLIC ${LLVM_INCLUDE_DIRS})
target_compile_definitions(TestIntegrationLegacyRuntimeDriverRunnable PUBLIC ${LLVM_DEFINITIONS})
