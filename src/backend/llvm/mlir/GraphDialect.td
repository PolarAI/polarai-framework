#ifndef GRAPH_OPS
#define GRAPH_OPS

include "mlir/IR/OpBase.td"

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
  let description = [{}];

  let arguments = (ins FlatSymbolRefAttr:$callee, Variadic<AnyTensor>:$inputs,
                    I64Attr:$tensor_addr, I64Attr:$node_id, StrAttr:$node_name,
                    I64Attr:$cluster_id);

  let results = (outs AnyTensor);

  // Add custom build methods for the generic call operation.
  let builders = [
    OpBuilder<"Builder *builder, OperationState &state, "
              "StringRef callee, ArrayRef<Value> arguments">
  ];
}

def ReturnOp : Graph_Op<"return", [Terminator, HasParent<"FuncOp">]> {
  let summary = "return operation";
  let description = [{}];

/*  let builders = [OpBuilder<
    "Builder *b, OperationState &state", [{ build(b, state, llvm::None); }]
  >];*/

  // Invoke a static verify method to verify this return operation.
  let verifier = [{ return ::verify(*this); }];
}

//===-------------------------------------------------------------------------===/
// Arithmetic operations
//===-------------------------------------------------------------------------===/

def AddOp : Graph_Op<"add", [NoSideEffect]> {
  let summary = "element-wise addition operation";
  let description = [{
    The "add" operation performs element-wise addition between two tensors.
    The shapes of the tensor operands are expected to match.
  }];

  let arguments = (ins AnyTensor:$lhs, AnyTensor:$rhs,
                   I64Attr:$tensor_addr, I64Attr:$node_id, StrAttr:$node_name,
                   I64Attr:$cluster_id);
  let results = (outs AnyTensor);

  let builders = [OpBuilder<"Builder *builder, OperationState &result, "
                                "const Value &a, "
                                "const Value &b, "
                                "const core::inner::Tensor &c, "
                                "int64_t node_id, StringRef node_name, int64_t cluster_id">];
}

def MulOp : Graph_Op<"mul", [NoSideEffect]> {
  let summary = "element-wise multiplication operation";
  let description = [{
    The "mul" operation performs element-wise addition between two tensors.
    The shapes of the tensor operands are expected to match.
  }];

  let arguments = (ins AnyTensor:$lhs, AnyTensor:$rhs,
                   I64Attr:$tensor_addr, I64Attr:$node_id, StrAttr:$node_name,
                   I64Attr:$cluster_id);
  let results = (outs AnyTensor);

  let builders = [OpBuilder<"Builder *builder, OperationState &result, "
                                  "const Value &a, "
                                  "const Value &b, "
                                  "const core::inner::Tensor &c, "
                                  "int64_t node_id, StringRef node_name, int64_t cluster_id">];
}

def MatmulOp : Graph_Op<"matmul", [NoSideEffect]> {
  let summary = "matrix-matrix multiplication operation";
  let description = [{
    The "mul" operation performs element-wise addition between two tensors.
    The shapes of the tensor operands are expected to match.
  }];

  let arguments = (ins 2DTensorOf<[AnyType]>:$lhs, 2DTensorOf<[AnyType]>:$rhs,
                   I64Attr:$tensor_addr, I64Attr:$node_id, StrAttr:$node_name,
                   I64Attr:$cluster_id);
  let results = (outs 2DTensorOf<[AnyType]>);

  let builders = [OpBuilder<"Builder *builder, OperationState &result, "
                                  "const Value &a, "
                                  "const Value &b, "
                                  "const core::inner::Tensor &c, "
                                  "int64_t node_id, StringRef node_name, int64_t cluster_id">];
}

//===-------------------------------------------------------------------------===/
// Memory operations
//===-------------------------------------------------------------------------===/

def AllocaOp : Graph_Op<"alloca", [NoSideEffect]> {
    let summary = "Allocate memory for tensor";

    let arguments = (ins
                     I64Attr:$tensor_addr, I64Attr:$node_id,
                     StrAttr:$node_name,
                     I64Attr:$cluster_id);
    let results = (outs AnyTensor);
    let builders = [OpBuilder<"Builder *builder, OperationState &result, "
                              "const core::inner::Tensor &tensor, "
                              "int64_t node_id, StringRef node_name, int64_t cluster_id">];
}

def MemlockOp : Graph_Op<"memlock"> {
    let summary = "Lock tensor data in memory";

    let arguments = (ins I64Attr:$tensor_addr, I64Attr:$node_id,
                     StrAttr:$node_name,
                     I64Attr:$cluster_id);
}

def MemreleaseOp : Graph_Op<"memrelease"> {
    let summary = "Unlock tensor data in memory";

    let arguments = (ins I64Attr:$tensor_addr, I64Attr:$node_id,
                     StrAttr:$node_name,
                     I64Attr:$cluster_id);
}

def TouchOp : Graph_Op<"touch"> {
    let summary = "Mark tensor as non-removable";

    let arguments = (ins I64Attr:$tensor_addr, I64Attr:$node_id,
                     StrAttr:$node_name,
                     I64Attr:$cluster_id);
}
#endif // TOY_OPS