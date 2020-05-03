#ifndef ATHENA_RUNTIME_OPS
#define ATHENA_RUNTIME_OPS

include "AthenaRuntimeDialect.td"
include "mlir/IR/OpAsmInterface.td"
include "mlir/Interfaces/CallInterfaces.td"
include "mlir/Interfaces/SideEffects.td"

def AthenaRuntime_LaunchOp : AthenaRuntime_Op<"launch", [CallOpInterface]> {
  let summary = "Launch kernel on device";
  let description = [{
    TBD
  }];

  let arguments = (ins FlatSymbolRefAttr:$kernel, Variadic<AnyType>:$operands);
  let results = (outs Variadic<AnyType>);

  let builders = [OpBuilder<
      "Builder *builder, OperationState &result, StringRef kernel, "
      "ValueRange operands = {}", [{
        result.addOperands(operands);
        result.addAttribute("kernel", builder->getSymbolRefAttr(kernel));
  }]>];

  let extraClassDeclaration = [{
      /// Get the argument operands to the called function.
      operand_range getArgOperands() {
        return {arg_operand_begin(), arg_operand_end()};
      }

      operand_iterator arg_operand_begin() { return operand_begin(); }
      operand_iterator arg_operand_end() { return operand_end(); }

      /// Return the callee of this operation.
      CallInterfaceCallable getCallableForCallee() {
        return getAttrOfType<SymbolRefAttr>("kernel");
      }
  }];

  let assemblyFormat = [{
    $kernel `(` $operands `)` attr-dict `:` functional-type($operands, results)
  }];
}

def AthenaRuntime_AnyCastOp : AthenaRuntime_Op<"any_cast", [NoSideEffect]> {
  let summary = "Casts any types";
  let description = [{
    TBD
  }];

  let arguments = (ins AnyType:$in);
  let results = (outs AnyType);
}

#endif // ATHENA_RUNTIME_OPS
