#ifndef ATHENA_GRAPH_OPS
#define ATHENA_GRAPH_OPS

include "AthenaGraphDialect.td"
include "mlir/IR/OpAsmInterface.td"
include "mlir/Interfaces/CallInterfaces.td"
include "mlir/Interfaces/ControlFlowInterfaces.td"
include "mlir/Interfaces/SideEffects.td"

def AthenaGraph_ModuleOp : AthenaGraph_Op<"module", [IsolatedFromAbove, SymbolTable,
                                            SingleBlockImplicitTerminator<"ModuleEndOp">]> {
  let summary = "A top level compilation unit containing host part of Athena Graph";
  let description = [{
    Athena Graph module contains code that describes data flow through execution graph.

    The module's top-level scope is modeled by a single region with a single block.
  }];

  // todo custom printer and parser

  let builders = [OpBuilder<"Builder *builder, OperationState &result">];

  let regions = (region SizedRegion<1>:$body);

  let skipDefaultBuilders = 1;
}

def AthenaGraph_ModuleEndOp : AthenaGraph_Op<"module_end", [Terminator, HasParent<"ModuleOp">]> {
  let summary = "A pseudo op that marks end of module";
  let description = [{
    This op terminates the only block of region in `ath_graph.module`.
  }];

  let parser = [{return success();}];
  let printer = [{ p << getOperationName(); }];
}

def AthenaGraph_NodeOp : AthenaGraph_Op<"node", [HasParent<"ModuleOp">, AutomaticAllocationScope,
                                                 FunctionLike, IsolatedFromAbove, Symbol]> {
  let summary = "Athena Graph Node";
  // todo extend description with samples when dialect is formed.
  let description = [{
    Defines a sequence of operations required to complete node execution. This includes allocation
    of tensors, locking tensors for particular access mode, performing operations on tensors.

    The operation has one region that corresponds to the body of node function.
  }];

  let regions = (region AnyRegion:$body);

  let skipDefaultBuilders = 1;

  let builders = [
    OpBuilder<"Builder *builder, OperationState &result, StringRef name, FunctionType type, "
              "ArrayRef<NamedAttribute> attrs = {}">
  ];

  let extraClassDeclaration = [{
    friend class OpTrait::FunctionLike<NodeOp>;
    unsigned getNumFuncArguments() { return getType().getNumInputs(); }
    unsigned getNumFuncResults() { return getType().getNumResults(); }
  }];
}

def AthenaGraph_GraphOp : AthenaGraph_Op<"graph", [HasParent<"ModuleOp">, AutomaticAllocationScope,
                                                   FunctionLike, IsolatedFromAbove, Symbol,
                                                   SingleBlockImplicitTerminator<"GraphTerminatorOp">]> {
  let summary = "Athena Graph operation";
  // todo extend description with samples when dialect is formed.
  let description = [{
  TBD
  }];

  let regions = (region AnyRegion:$body);

  let skipDefaultBuilders = 1;

  let builders = [
    OpBuilder<"Builder *builder, OperationState &result, StringRef name, "
              "ArrayRef<NamedAttribute> attrs = {}">
  ];

  let extraClassDeclaration = [{
    friend class OpTrait::FunctionLike<GraphOp>;
    unsigned getNumFuncArguments() { return getType().getNumInputs(); }
    unsigned getNumFuncResults() { return getType().getNumResults(); }
  }];
}

def AthenaGraph_ReturnOp : AthenaGraph_Op<"return", [NoSideEffect, HasParent<"NodeOp">, Terminator
                                                     /*ReturnLike*/]> {
  let summary = "return operation";
  let description = [{
    Returns result of Athena Graph Node computation.
  }];

  // fixme should this be AnyTensor?
  let arguments = (ins Variadic<AnyType>:$operands);
  let builders = [OpBuilder<
    "Builder *b, OperationState &result", [{ build(b, result, llvm::None); }]
  >];
  let assemblyFormat = "attr-dict ($operands^ `:` type($operands))?";
}

def AthenaGraph_GraphTerminatorOp : AthenaGraph_Op<"graph_terminator", [Terminator,
      HasParent<"GraphOp">, NoSideEffect]> {
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
      "Builder *builder, OperationState &result, NodeOp node,"
      "ValueRange operands = {}", [{
        result.addOperands(operands);
        result.addAttribute("node", builder->getSymbolRefAttr(node));
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

  let builders = [OpBuilder<"Builder *builder, OperationState &result, Value slice, Value tensor">];
  let skipDefaultBuilders = 1;
}

def AthenaGraph_GetTensor : AthenaGraph_Op<"get_tensor", [NoSideEffect]> {
  let summary = "Get tensor by virtual address";
  let description = [{ TBD }];

  let arguments = (ins AnyType:$context, IndexAttr:$virtual_address);
  let results = (outs AnyTensor);

  let builders = [OpBuilder<"Builder *builder, OperationState &result, Value context, "
                            "size_t virtualAddress, RankedTensorType type">];
  let skipDefaultBuilders = 1;
}

//===------------------------------------------------------------------------------------------===//
// Athena Graph builtins
//===------------------------------------------------------------------------------------------===//

def AthenaGraph_AddOp : AthenaGraph_Op<"add", [NoSideEffect]> {
  let summary = "Element-wise addition operation";
  let description = [{ TBD }];

  let arguments = (ins AnyTensor:$a, AnyType:$scaleA, AnyTensor:$b, AnyType:$scaleB, AnyTensor:$out);
  let results = (outs AnyTensor);
}

def AthenaGraph_MulOp : AthenaGraph_Op<"mul", [NoSideEffect]> {
  let summary = "Element-wise multiplication operation";
  let description = [{ TBD }];

  let arguments = (ins AnyTensor:$a, AnyType:$scaleA, AnyTensor:$b, AnyType:$scaleB, AnyTensor:$out);
  let results = (outs AnyTensor);
}

def AthenaGraph_MatmulOp : AthenaGraph_Op<"matmul", [NoSideEffect]> {
  let summary = "Matrix-matrix multiplication operation";
  let description = [{ TBD }];

  let arguments = (ins AnyTensor:$a, AnyType:$scaleA, AnyTensor:$b, AnyType:$scaleB, AnyTensor:$out);
  let results = (outs AnyTensor);
}

def AthenaGraph_TransposeOp : AthenaGraph_Op<"transpose", [NoSideEffect]> {
  let summary = "Matrix transposition operation";
  let description = [{ TBD }];

  let arguments = (ins AnyTensor:$a, AnyTensor:$res);
  let results = (outs AnyTensor);
}

def AthenaGraph_FillOp : AthenaGraph_Op<"fill", [NoSideEffect]> {
  let summary = "Fills tensor with value";
  let description = [{ TBD }];

  let arguments = (ins AnyType:$pattern, AnyTensor:$out);
  let results = (outs AnyTensor);
}

def AthenaGraph_AllocOp : AthenaGraph_Op<"alloc", [NoSideEffect]> {
  let summary = "Allocates memory for tensor";
  let description = [{}];

  let arguments = (ins AnyTensor:$target);
}

def AthenaGraph_LockOp : AthenaGraph_Op<"lock"> {
  let summary = "Locks data on device";
  let description = [{
    Athena LLVM backend Allocators explicitly manage memory both on host and devices. Memory can be
    moved from one domain to another. To prevent this, tensors that are about to be accessed must be
    "locked" on a certain device.

    Nodes can lock tensors for read and read_write access. Multiple devices are allowed to lock
    tensor for read, and only one device is allowed to lock tensor for read_write.
  }];

  // todo change to enum.
  let arguments = (ins StrAttr:$lockType, AnyTensor:$lockedTensor);
}

def AthenaGraph_ReleaseOp : AthenaGraph_Op<"release"> {
  let summary = "Releases data";
  let description = [{
    Notifies Athena LLVM backend Allocator, that tensor is no longer in use and its memory can be
    freed.
  }];

  let arguments = (ins AnyTensor:$tensor);
}

def AthenaGraph_InvokeLoaderOp : AthenaGraph_Op<"invoke_loader"> {
  let summary = "Invokes loader";
  let description = [{
    One way of getting data into tensor is invoking a loader routine. Users are allowed to write
    their own loaders. Loader must provide a C-like function that can be invoked on host.
  }];

  let arguments = (ins StrAttr:$loader_routine, AnyTensor:$dest);

}
#endif // ATHENA_GRAPH_OPS
