file(GLOB CORE_TEST_SRC "*.cpp")

find_package(LLVM CONFIG)
llvm_map_components_to_libnames(llvm_libs core irreader support transformutils passes codegen orcjit x86asmparser x86codegen lto)

add_athena_executable(runUnitLLVMBackend ${CORE_TEST_SRC})
target_link_libraries(runUnitLLVMBackend athena-core backend-llvm)
target_link_libraries(runUnitLLVMBackend ${GTEST_BOTH_LIBRARIES} Threads::Threads)
target_include_directories(runUnitLLVMBackend PRIVATE ${GTEST_INCLUDE_DIRS})

target_include_directories(runUnitLLVMBackend PRIVATE ${LLVM_INCLUDE_DIRS})
target_compile_definitions(runUnitLLVMBackend PRIVATE ${LLVM_DEFINITIONS})

add_test(NAME UnitLLVMBackendTests COMMAND runUnitLLVMBackend)

install(TARGETS runUnitLLVMBackend
        RUNTIME DESTINATION test)
