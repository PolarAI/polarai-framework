#ifndef GRAPH_OPS
#define GRAPH_OPS

include "mlir/IR/OpBase.td"
include "mlir/Interfaces/SideEffects.td"

def Graph_Dialect : Dialect {
  let name = "graph";

  // The C++ namespace that the dialect class definition resides in.
  let cppNamespace = "athena::backend::llvm";
}

class Graph_Op<string mnemonic, list<OpTrait> traits = []> :
    Op<Graph_Dialect, mnemonic, traits>;

//===-------------------------------------------------------------------------===/
// Control Flow operations
//===-------------------------------------------------------------------------===/

def CallOp : Graph_Op<"call"> {
  let summary = "call operation";
  let description = [{
    The `graph.call` operation allows one to make calls to arbitrary functions outside the graph.
    This is useful for loader invocation and user-defined operations.
  }];

  let arguments = (ins FlatSymbolRefAttr:$callee, Variadic<AnyTensor>:$inputs,
                    I64Attr:$tensor_addr, I64Attr:$node_id, StrAttr:$node_name,
                    I64Attr:$cluster_id);

  let results = (outs AnyTensor);

  // Add custom build methods for the generic call operation.
  let builders = [
    OpBuilder<"OpBuilder &builder, OperationState &state, "
              "StringRef callee, ArrayRef<Value> arguments">
  ];
}

def ReturnOp : Graph_Op<"return", [Terminator, HasParent<"FuncOp">]> {
  let summary = "return operation";
  let description = [{
    The `graph.return` operation denotes the end of the graph.
  }];

  // Invoke a static verify method to verify this return operation.
  let verifier = [{ return ::verify(*this); }];
}

//===-------------------------------------------------------------------------===/
// Arithmetic operations
//===-------------------------------------------------------------------------===/

def AddOp : Graph_Op<"add", [NoSideEffect]> {
  let summary = "element-wise addition operation";
  let description = [{
    The `graph.add` operation performs element-wise addition between two tensors.
    The shapes of the tensor operands are expected to match.
  }];

  let arguments = (ins AnyTensor:$lhs, AnyTensor:$rhs,
                   I64Attr:$tensor_addr, I64Attr:$node_id, StrAttr:$node_name,
                   I64Attr:$cluster_id);
  let results = (outs AnyTensor);

  let builders = [OpBuilder<"OpBuilder &builder, OperationState &result, "
                                "const Value &a, "
                                "const Value &b, "
                                "const core::inner::Tensor &c, "
                                "int64_t node_id, StringRef node_name, int64_t cluster_id">];
}

def MulOp : Graph_Op<"mul", [NoSideEffect]> {
  let summary = "element-wise multiplication operation";
  let description = [{
    The `graph.mul` operation performs element-wise multiplication between two tensors.
    The shapes of the tensor operands are expected to match.
  }];

  let arguments = (ins AnyTensor:$lhs, AnyTensor:$rhs,
                   I64Attr:$tensor_addr, I64Attr:$node_id, StrAttr:$node_name,
                   I64Attr:$cluster_id);
  let results = (outs AnyTensor);

  let builders = [OpBuilder<"OpBuilder &builder, OperationState &result, "
                                  "const Value &a, "
                                  "const Value &b, "
                                  "const core::inner::Tensor &c, "
                                  "int64_t node_id, StringRef node_name, int64_t cluster_id">];
}

def MatmulOp : Graph_Op<"matmul", [NoSideEffect]> {
  let summary = "generic matrix-matrix multiplication operation";
  let description = [{
    The `graph.matmul` operation performs generic matrix-matrix multiplication of tensors.
    The operands are expected to be rank 2 tensor with shapes, that allow such operation.

    ```
    C = alpha * A * B + beta * C
    ```
  }];

  let arguments = (ins 2DTensorOf<[AnyType]>:$lhs, 2DTensorOf<[AnyType]>:$rhs,
                   BoolAttr:$transposeA, BoolAttr:$transposeB,
                   AnyType:$alpha, AnyType:$beta,
                   I64Attr:$tensor_addr, I64Attr:$node_id, StrAttr:$node_name,
                   I64Attr:$cluster_id);
  let results = (outs 2DTensorOf<[AnyType]>);

  let builders = [OpBuilder<"OpBuilder &builder, OperationState &result, "
                                  "const Value &a, "
                                  "const Value &b, "
                                  "bool transposeA, bool transposeB, "
                                  "uint64_t alpha, uint64_t beta, "
                                  "const core::inner::Tensor &c, "
                                  "int64_t node_id, StringRef node_name, int64_t cluster_id">];
}

//===-------------------------------------------------------------------------===/
// Memory operations
//===-------------------------------------------------------------------------===/

def AllocaOp : Graph_Op<"alloca", [NoSideEffect]> {
    let summary = "memory allocation operation";

    let description = [{
      The `graph.alloca` operation performs memory allocation for tensor.
    }];

    let arguments = (ins
                     I64Attr:$tensor_addr, I64Attr:$node_id,
                     StrAttr:$node_name,
                     I64Attr:$cluster_id);
    let results = (outs AnyTensor);
    let builders = [OpBuilder<"OpBuilder &builder, OperationState &result, "
                              "const core::inner::Tensor &tensor, "
                              "int64_t node_id, StringRef node_name, int64_t cluster_id">];
}

def MemlockOp : Graph_Op<"memlock"> {
    let summary = "memory lock operation";

    let description = [{
      The `graph.memlock` operation locks tensor in device memory
      preventing it from being forced out by other memory operations.
    }];

    let arguments = (ins I64Attr:$tensor_addr, I64Attr:$node_id,
                     StrAttr:$node_name,
                     I64Attr:$cluster_id);
}

def MemreleaseOp : Graph_Op<"memrelease"> {
    let summary = "memory release operation";

    let description = [{
      The `graph.memrelease` operation marks memory region as free.
    }];

    let arguments = (ins I64Attr:$tensor_addr, I64Attr:$node_id,
                     StrAttr:$node_name,
                     I64Attr:$cluster_id);
}

def TouchOp : Graph_Op<"touch"> {
    let summary = "Mark tensor as non-removable";

    let arguments = (ins AnyTensor:$tensor, I64Attr:$node_id,
                     StrAttr:$node_name,
                     I64Attr:$cluster_id);
}
#endif // TOY_OPS