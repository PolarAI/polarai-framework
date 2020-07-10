get_property(dialect_libs GLOBAL PROPERTY MLIR_DIALECT_LIBS)
get_property(conversion_libs GLOBAL PROPERTY MLIR_CONVERSION_LIBS)
set(LIBS
        MLIRPolarRuntime
        MLIRPolarGraph
        MLIRCompute
        ${dialect_libs}
        ${conversion_libs}
        MLIROptLib
        MLIRPolarConversion
        MLIRPolarPasses
        )
add_llvm_executable(polar-opt main.cpp)

llvm_update_compile_flags(polar-opt)
target_link_libraries(polar-opt PRIVATE ${LIBS})
