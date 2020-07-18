set(LLVM_TARGET_DEFINITIONS PolarGraphOps.td)
add_mlir_dialect(PolarGraphOps polar_graph)
mlir_tablegen(PolarGraphOpsEnums.h.inc -gen-enum-decls)
mlir_tablegen(PolarGraphOpsEnums.cpp.inc -gen-enum-defs)

add_mlir_interface(ComputationalOpInterface)

add_mlir_doc(PolarGraphDialect -gen-dialect-doc PolarGraphDialect PolarGraph/)
add_mlir_doc(PolarGraphOps -gen-op-doc PolarGraphOps PolarGraph/)
