find_package(LLVM CONFIG)

llvm_map_components_to_libnames(llvm_libs
        core
        support
        tablegen)

add_athena_executable(athena-tablegen main.cpp)
target_link_libraries(athena-tablegen PRIVATE ${llvm_libs} athena_tblgen_backends)
target_include_directories(athena-tablegen PRIVATE ${LLVM_INCLUDE_DIRS})
target_compile_definitions(athena-tablegen PRIVATE ${LLVM_DEFINITIONS})