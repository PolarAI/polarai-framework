find_package(LLVM CONFIG)

llvm_map_components_to_libnames(llvm_libs
        core
        support
        tablegen)

add_library(athena_tblgen_backends STATIC MarkdownBackend.cpp RuntimeInfoBackend.cpp WrapperBackend.cpp)
target_link_libraries(athena_tblgen_backends PRIVATE ${llvm_libs})
target_include_directories(athena_tblgen_backends PRIVATE ${LLVM_INCLUDE_DIRS})
target_compile_definitions(athena_tblgen_backends PRIVATE ${LLVM_DEFINITIONS})