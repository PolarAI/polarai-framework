get_property(dialect_libs GLOBAL PROPERTY MLIR_DIALECT_LIBS)
get_property(conversion_libs GLOBAL PROPERTY MLIR_CONVERSION_LIBS)
set(LIBS
        ${dialect_libs}
        ${conversion_libs}
        MLIROptLib
        MLIRAthenaConversion
        MLIRAthenaGraph
        MLIRAthenaPasses
        )
add_llvm_executable(athena-opt main.cpp)

llvm_update_compile_flags(athena-opt)
target_link_libraries(athena-opt PRIVATE ${LIBS})