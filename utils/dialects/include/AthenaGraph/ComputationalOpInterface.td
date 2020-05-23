#ifndef ATHENA_COMPUTATIONAL_OP_INTERFACE
#define ATHENA_COMPUTATIONAL_OP_INTERFACE

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
    }]>
  ];
}

def Computational : DeclareOpInterfaceMethods<ComputationalOpInterface>;

#endif // ATHENA_COMPUTATIONAL_OP_INTERFACE