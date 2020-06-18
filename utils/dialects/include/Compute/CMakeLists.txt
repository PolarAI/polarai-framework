set(LLVM_TARGET_DEFINITIONS ComputeInterfaces.td)
mlir_tablegen(ComputeInterfaces.h.inc -gen-op-interface-decls)
mlir_tablegen(ComputeOpInterfaces.cpp.inc -gen-op-interface-defs)
add_public_tablegen_target(MLIRComputeInterfacesIncGen)

add_mlir_dialect(ComputeOps compute)
add_mlir_doc(ComputeDialect -gen-dialect-doc ComputeDialect Compute/)
add_mlir_doc(ComputeOps -gen-op-doc ComputeOps Compute/)
