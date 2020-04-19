find_package(LLVM CONFIG)
llvm_map_components_to_libnames(llvm_libs support)

find_package(OpenCL)
if (OpenCL_FOUND)
add_athena_integration_test(
        TARGET_NAME OpenCLRuntime
        SRCS main.cpp
        LIBS athena ${llvm_libs} OpenCL::OpenCL
)
add_dependencies(TestIntegrationOpenCLRuntimeRunnable ${ATH_RT_LLVM_OCL})

target_include_directories(TestIntegrationOpenCLRuntimeRunnable PUBLIC ${LLVM_INCLUDE_DIRS})
target_compile_definitions(TestIntegrationOpenCLRuntimeRunnable PUBLIC ${LLVM_DEFINITIONS})
endif()
