file(GLOB CORE_TEST_SRC "*.cpp")

find_package(LLVM CONFIG)
llvm_map_components_to_libnames(llvm_libs core irreader support transformutils passes codegen orcjit x86asmparser x86codegen lto)

add_athena_executable(TestUnitBackendLLVMRunnable ${CORE_TEST_SRC})
target_link_libraries(TestUnitBackendLLVMRunnable PRIVATE
        athena-core
        backend-llvm
        Threads::Threads
        AthenaDep::googletest)

target_include_directories(TestUnitBackendLLVMRunnable PRIVATE ${LLVM_INCLUDE_DIRS})
target_compile_definitions(TestUnitBackendLLVMRunnable PRIVATE ${LLVM_DEFINITIONS})

add_test(NAME BackendLLVMUnitTest COMMAND TestUnitBackendLLVMRunnable)
