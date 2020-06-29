#ifndef POLAR_COMPUTATIONAL_OP_INTERFACE
#define POLAR_COMPUTATIONAL_OP_INTERFACE

include "mlir/IR/OpBase.td"

def ComputationalOpInterface : OpInterface<"ComputationalOpInterface"> {
  let description = [{
    Marks Athena Graph operations that perform actual computation. Such
    operations are lowered to `ath_rt.launch` calls.
  }];
  let methods = [
    InterfaceMethod<"Get kernel name based on operation name and operands",
    "llvm::StringRef", "getKernelName", (ins), [{}], [{
      return "dummy";
    }]>,
    InterfaceMethod<"Produces Compute kernel to calculate operation",
      "void", "produceKernel", (ins "OpBuilder&":$builder,
      "Block::BlockArgListType":$args), [{}], [{}]>
  ];
}

def Computational : DeclareOpInterfaceMethods<ComputationalOpInterface>;

#endif // POLAR_COMPUTATIONAL_OP_INTERFACE
