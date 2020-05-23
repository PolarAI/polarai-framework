add_mlir_dialect(AthenaGraphOps ath_graph)

# FIXME uncomment when build is updated
#add_mlir_interface(ComputationalOpInterface)

set(LLVM_TARGET_DEFINITIONS ComputationalOpInterface.td)
mlir_tablegen(ComputationalOpInterface.h.inc -gen-op-interface-decls)
mlir_tablegen(ComputationalOpInterface.cpp.inc -gen-op-interface-defs)
add_public_tablegen_target(MLIRComputationalOpInterfaceIncGen)

add_mlir_doc(AthenaGraphDialect -gen-dialect-doc AthenaGraphDialect AthenaGraph/)
add_mlir_doc(AthenaGraphOps -gen-op-doc AthenaGraphOps AthenaGraph/)
