#ifndef ATHENA_GRAPH_OPS
#define ATHENA_GRAPH_OPS

include "AthenaGraphDialect.td"
include "mlir/IR/OpAsmInterface.td"
include "mlir/IR/SymbolInterfaces.td"
include "mlir/Interfaces/CallInterfaces.td"
include "mlir/Interfaces/ControlFlowInterfaces.td"
include "mlir/Interfaces/SideEffects.td"

def AthenaGraph_NodeOp : AthenaGraph_Op<"node", [AutomaticAllocationScope,
                                                 FunctionLike, 
                                                 IsolatedFromAbove, Symbol]> {
  let summary = "Athena Graph Node";
  // todo extend description with samples when dialect is formed.
  let description = [{
    Defines a sequence of operations required to complete node execution. 
    This includes allocation of tensors, locking tensors for particular access 
    mode, performing operations on tensors.

    The operation has one region that corresponds to the body of node function.
  }];

  let regions = (region AnyRegion:$body);

  let skipDefaultBuilders = 1;

  let builders = [
    OpBuilder<"OpBuilder& builder, OperationState &result, StringRef name, "
              "FunctionType type, size_t nodeId, size_t clusterId, "
              "ArrayRef<NamedAttribute> attrs = {}">
  ];

  let extraClassDeclaration = [{
    friend class OpTrait::FunctionLike<NodeOp>;
    unsigned getNumFuncArguments() { return getType().getNumInputs(); }
    unsigned getNumFuncResults() { return getType().getNumResults(); }
    static StringRef getNodeIdAttrName() {
      return "node_id";
    }
    static StringRef getClusterIdAttrName() {
      return "cluster_id";
    }
  }];
  // let printer = [{ printNodeOp(p, *this); }];
  // let parser = [{ return parseNodeOp(parser, result); }];
}

def AthenaGraph_GraphOp : AthenaGraph_Op<"graph", 
                          [AutomaticAllocationScope, FunctionLike, 
                          IsolatedFromAbove, Symbol,
                          SingleBlockImplicitTerminator<"GraphTerminatorOp">]> {
  let summary = "Athena Graph operation";
  // todo extend description with samples when dialect is formed.
  let description = [{
  TBD
  }];

  let regions = (region AnyRegion:$body);

  let skipDefaultBuilders = 1;

  let builders = [
    OpBuilder<"OpBuilder& builder, OperationState &result, StringRef name, "
              "ArrayRef<NamedAttribute> attrs = {}">
  ];

  let extraClassDeclaration = [{
    friend class OpTrait::FunctionLike<GraphOp>;
    unsigned getNumFuncArguments() { return getType().getNumInputs(); }
    unsigned getNumFuncResults() { return getType().getNumResults(); }
  }];
}

def AthenaGraph_ReturnOp : AthenaGraph_Op<"return", [NoSideEffect, 
                                                     HasParent<"NodeOp">,
                                                     Terminator
                                                     /*ReturnLike*/]> {
  let summary = "return operation";
  let description = [{
    Returns result of Athena Graph Node computation.
  }];

  // fixme should this be AnyTensor?
  let arguments = (ins Variadic<AnyType>:$operands);
  let builders = [OpBuilder<
    "OpBuilder& b, OperationState &result", [{ build(b, result, llvm::None); }]
  >];
  let assemblyFormat = "attr-dict ($operands^ `:` type($operands))?";
}

def AthenaGraph_GraphTerminatorOp : AthenaGraph_Op<"graph_terminator", 
                                    [Terminator, HasParent<"GraphOp">, 
                                    NoSideEffect]> {
  let summary = "graph terminator operation";
  let description = [{
    Indicates end of graph computation.
  }];
}

def AthenaGraph_EvalOp : AthenaGraph_Op<"eval", [CallOpInterface]> {
  let summary = "Evaluate graph node";
  let description = [{ TBD }];

  let arguments = (ins FlatSymbolRefAttr:$node, Variadic<AnyType>:$operands);
  let results = (outs Variadic<AnyType>);

  let builders = [OpBuilder<
      "OpBuilder& builder, OperationState &result, NodeOp node,"
      "ValueRange operands = {}", [{
        result.addOperands(operands);
        result.addAttribute("node", builder.getSymbolRefAttr(node));
        result.addTypes(node.getType().getResults());
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
        return getAttrOfType<SymbolRefAttr>("node");
      }
  }];

  let assemblyFormat = [{
    $node `(` $operands `)` attr-dict `:` functional-type($operands, results)
  }];
}

def AthenaGraph_SliceOp : AthenaGraph_Op<"slice", [NoSideEffect]> {
  let summary = "Get sub-tensor by first dimension offset";
  let description = [{ TBD }];

  let arguments = (ins AnyType:$id, AnyTensor:$tensor);
  let results = (outs AnyTensor);

  let builders = [OpBuilder<"OpBuilder& builder, OperationState &result, "
                            "Value slice, Value tensor">];
  let skipDefaultBuilders = 1;
}

def AthenaGraph_CreateTensorOp : AthenaGraph_Op<"create_tensor", 
                                                [NoSideEffect]> {
  let summary = "Get tensor by virtual address";
  let description = [{ TBD }];

  let arguments = (ins IndexAttr:$virtual_address);
  let results = (outs AnyTensor);

  let builders = [OpBuilder<"OpBuilder& builder, OperationState &result, "
                            "uint64_t virtualAddress, RankedTensorType type">];
  let skipDefaultBuilders = 1;
  let extraClassDeclaration = [{
    static llvm::StringRef getVirtualAddressAttrName() {
      return "virtual_address";
    }
  }];
}

def AthenaGraph_AllocOp : AthenaGraph_Op<"alloc"> {
  let summary = "Allocates memory for tensor";
  let description = [{}];

  let arguments = (ins AnyTensor:$target);
}

def AthenaGraph_LockOp : AthenaGraph_Op<"lock"> {
  let summary = "Locks data on device";
  let description = [{
    Athena LLVM backend Allocators explicitly manage memory both on host and 
    devices. Memory can be moved from one domain to another. To prevent this, 
    tensors that are about to be accessed must be "locked" on a certain device.

    Nodes can lock tensors for read and read_write access. Multiple devices are
    allowed to lock tensor for read, and only one device is allowed to lock
    tensor for read_write.
  }];

  // todo change to enum.
  let arguments = (ins StrAttr:$lock_type, AnyTensor:$locked_tensor);
}

def AthenaGraph_ReleaseOp : AthenaGraph_Op<"release"> {
  let summary = "Releases data";
  let description = [{
    Notifies Athena LLVM backend Allocator, that tensor is no longer in use and
    its memory can be freed.
  }];

  let arguments = (ins AnyTensor:$tensor);
}

def AthenaGraph_InvokeLoaderOp : AthenaGraph_Op<"invoke_loader"> {
  let summary = "Invokes loader";
  let description = [{
    One way of getting data into tensor is invoking a loader routine. Users are
    allowed to write their own loaders. Loader must provide a C-like function
    that can be invoked on host.
  }];

  let arguments = (ins AnyTensor:$dest);

}

def AthenaGraph_BarrierOp : AthenaGraph_Op<"barrier", [IsolatedFromAbove]> {
  let summary = "Blocks execution of the subsequent nodes";
  let description = [{ TBD }];

  let arguments = (ins IndexAttr:$clusterId);
}

//===----------------------------------------------------------------------===//
// Athena Graph builtins
//===----------------------------------------------------------------------===//

def AthenaGraph_AddOp : AthenaGraph_Op<"add", [ComputationalOpInterface]> {
  let summary = "Element-wise addition operation";
  let description = [{ TBD }];

  let arguments = (ins AnyTensor:$a, AnyType:$scaleA, AnyTensor:$b,
                       AnyType:$scaleB, AnyType:$size, AnyTensor:$out);
  let results = (outs AnyTensor);
  let builders = [OpBuilder<"OpBuilder& builder, OperationState& result, "
                            "Value a, Value scaleA, Value b, Value scaleB, Value size, "
                            "Value out", [{
    result.addOperands(a);
    result.addOperands(scaleA);
    result.addOperands(b);
    result.addOperands(scaleB);
    result.addOperands(size);
    result.addOperands(out);
    result.types.push_back(out.getType());
  }]>];
  let extraClassDeclaration = [{
    llvm::StringRef getKernelName() {
      return "fadd";
    }
  }];
}

def AthenaGraph_CopyOp : AthenaGraph_Op<"copy", [ComputationalOpInterface]> {
  let summary = "Element-wise copy operation";
  let description = [{ TBD }];

  let arguments = (ins AnyTensor:$input, AnyType:$size, AnyTensor:$out);
  let results = (outs AnyTensor);
  let builders = [OpBuilder<"OpBuilder& builder, OperationState& result, "
                            "Value input, Value size, "
                            "Value out", [{
    result.addOperands(input);
    result.addOperands(size);
    result.addOperands(out);
    result.types.push_back(out.getType());
  }]>];
  let extraClassDeclaration = [{
    llvm::StringRef getKernelName() {
      return "fcopy";
    }
  }];
}

def AthenaGraph_DivideOp : AthenaGraph_Op<"divide", [ComputationalOpInterface]> {
  let summary = "Element-wise division operation";
  let description = [{ TBD }];

  let arguments = (ins AnyTensor:$numerator, AnyTensor:$denominator, AnyType:$size,
  AnyTensor:$out);
  let results = (outs AnyTensor);
  let builders = [OpBuilder<"OpBuilder& builder, OperationState& result, "
                            "Value numerator, Value denominator, Value size, "
                            "Value out", [{
    result.addOperands(numerator);
    result.addOperands(denominator);
    result.addOperands(size);
    result.addOperands(out);
    result.types.push_back(out.getType());
  }]>];
  let extraClassDeclaration = [{
    llvm::StringRef getKernelName() {
      return "fdivide";
    }
  }];
}

def AthenaGraph_LogLossOp : AthenaGraph_Op<"logloss", [ComputationalOpInterface]> {
  let summary = "Element-wise logistic loss operation";
  let description = [{ TBD }];

  let arguments = (ins AnyTensor:$predicted, AnyTensor:$groundTruth, AnyType:$size,
  AnyTensor:$out);
  let results = (outs AnyTensor);
  let builders = [OpBuilder<"OpBuilder& builder, OperationState& result, "
                            "Value predicted, Value groundTruth, Value size, "
                            "Value out", [{
    result.addOperands(predicted);
    result.addOperands(groundTruth);
    result.addOperands(size);
    result.addOperands(out);
    result.types.push_back(out.getType());
  }]>];
  let extraClassDeclaration = [{
    llvm::StringRef getKernelName() {
      return "flogloss";
    }
  }];
}

def AthenaGraph_MatMulOp : AthenaGraph_Op<"matmul", [ComputationalOpInterface]> {
  let summary = "Matrix-matrix multiplication operation";
  let description = [{ TBD }];

  let arguments = (ins AnyType:$transLeft, AnyType:$transRight, AnyType:$m,
  AnyType:$n, AnyType:$k, AnyTensor:$left, AnyTensor:$right, AnyTensor:$out);
  let results = (outs AnyTensor);
  let builders = [OpBuilder<"OpBuilder& builder, OperationState& result, "
                            "Value transLeft, Value transRight, Value m, Value n, Value k, Value left, Value right, "
                            "Value out", [{
  result.addOperands(transLeft);
  result.addOperands(transRight);
  result.addOperands(m);
  result.addOperands(n);
  result.addOperands(k);
  result.addOperands(left);
  result.addOperands(right);
  result.addOperands(out);
  result.types.push_back(out.getType());
  }]>];
  let extraClassDeclaration = [{
    llvm::StringRef getKernelName() {
      return "fmatmul";
    }
  }];
}

def AthenaGraph_MulOp : AthenaGraph_Op<"mul", [ComputationalOpInterface]> {
  let summary = "Element-wise multiplication operation";
  let description = [{ TBD }];

  let arguments = (ins AnyTensor:$a, AnyTensor:$b, AnyType:$size,
                       AnyTensor:$out);
  let results = (outs AnyTensor);
  let builders = [OpBuilder<"OpBuilder& builder, OperationState& result, "
                            "Value a, Value b, Value size, "
                            "Value out", [{
    result.addOperands(a);
    result.addOperands(b);
    result.addOperands(size);
    result.addOperands(out);
    result.types.push_back(out.getType());
  }]>];
  let extraClassDeclaration = [{
    llvm::StringRef getKernelName() {
      return "fmul";
    }
  }];
}

def AthenaGraph_MulConcatOp : AthenaGraph_Op<"mulconcat", [ComputationalOpInterface]> {
  let summary = "Element-wise multiplication operation";
  let description = [{ TBD }];

  let arguments = (ins AnyTensor:$gradient, AnyType:$gradientSize, AnyTensor:$localDerivative, AnyType:$localDerivativeSize,
  AnyTensor:$out);
  let results = (outs AnyTensor);
  let builders = [OpBuilder<"OpBuilder& builder, OperationState& result, "
                            "Value gradient, Value gradientSize, Value localDerivative, Value localDerivativeSize, "
                            "Value out", [{
    result.addOperands(gradient);
    result.addOperands(gradientSize);
    result.addOperands(localDerivative);
    result.addOperands(localDerivativeSize);
    result.addOperands(out);
    result.types.push_back(out.getType());
  }]>];
  let extraClassDeclaration = [{
    llvm::StringRef getKernelName() {
      return "fmulconcat";
    }
  }];
}

def AthenaGraph_SigmoidOp : AthenaGraph_Op<"sigmoid", [ComputationalOpInterface]> {
  let summary = "Element-wise sigmoid operation";
  let description = [{ TBD }];

  let arguments = (ins AnyTensor:$input, AnyType:$size,
  AnyTensor:$out);
  let results = (outs AnyTensor);
  let builders = [OpBuilder<"OpBuilder& builder, OperationState& result, "
                            "Value input, Value size, "
                            "Value out", [{
    result.addOperands(input);
    result.addOperands(size);
    result.addOperands(out);
    result.types.push_back(out.getType());
  }]>];
  let extraClassDeclaration = [{
    llvm::StringRef getKernelName() {
      return "fsigmoid";
    }
  }];
}


def AthenaGraph_TransposeOp : AthenaGraph_Op<"transpose", [Computational]> {
  let summary = "Matrix transposition operation";
  let description = [{ TBD }];

  let arguments = (ins AnyTensor:$a, AnyTensor:$size, AnyTensor:$res);
  let results = (outs AnyTensor);
  let builders = [OpBuilder<"OpBuilder& builder, OperationState& result, "
                            "Value a, Value size, Value out", [{
    result.addOperands(a);
    result.addOperands(size);
    result.addOperands(out);
  result.types.push_back(out.getType());
  }]>];
}

def AthenaGraph_FillOp : AthenaGraph_Op<"fill", [ComputationalOpInterface]> {
  let summary = "Fills tensor with value";
  let description = [{ TBD }];

  let arguments = (ins AnyType:$pattern, AnyType:$size, AnyTensor:$out);
  let results = (outs AnyTensor);
  let builders = [OpBuilder<"OpBuilder& builder, OperationState& result, "
                            "Value pattern, Value size, Value out", [{
    result.addOperands(pattern);
    result.addOperands(size);
    result.addOperands(out);
    result.types.push_back(out.getType());
  }]>];
  let extraClassDeclaration = [{
    llvm::StringRef getKernelName() {
      return "ffill";
    }
  }];
}

#endif // ATHENA_GRAPH_OPS
