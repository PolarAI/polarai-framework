#ifndef COMPUTE_INTERFACES
#define COMPUTE_INTERFACES

include "mlir/IR/OpBase.td"

def LaunchInterface : OpInterface<"LaunchInterface"> {
  let description = [{
    Marks operations that are capable of launching Compute kernels.
  }];
  let methods = [
    InterfaceMethod<"Get kernel name",
    "llvm::StringRef", "getKernelName", (ins)>,
    InterfaceMethod<"Get containing module",
    "SymbolRefAttr", "getModule", (ins)>
  ];
}
#endif // COMPUTE_INTERFACES 
