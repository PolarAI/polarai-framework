---
id: mlir_graph_deialect
title: MLIR Graph dialect
---

# Rationale

While LLVM IR is good for representing low-level code for further machine 
instruction generation, performing high-level optimizations is complicated.
On the other hand, MLIR is designed to be extensible intermediate representation.
To learn more visit https://mlir.llvm.org.

# Dialect specification

## High-level structure

Every Graph in the dialect is represented with a function. Every instruction of
this function corresponds to one of the nodes. Node and cluster are specified by
mandatory attributes `node_id`, `node_name` and `cluster_id`. These can be omitted 
only for `graph.return` operation.

Here's an example of a simple graph:

```mlir
module {
    func @add() {
        %arg0 = "graph.alloca"() { tensor_addr = 1, node_id = 1, node_name = "inputA", cluster_id = 0} : () -> tensor<3xf32>
        "graph.call"(%arg0, 0x0001) { callee = @MemoryLoaderLoad, node_id = 1, node_name = "inputA", cluster_id = 0} : () -> ()
        %arg1 = "graph.alloca"() { tensor_addr = 4, node_id = 2, node_name = "inputB", cluster_id = 0} : () -> tensor<3xf32>
        "graph.call"(%arg1, 0x0020) { callee = @MemoryLoaderLoad, node_name = "inputB", cluster_id = 0} : () -> ()
        %res = "graph.alloca"() { tensor_addr = 7, node_id = 3, node_name = "inputC", cluster_id = 0} : () -> tensor<3xf32>
        "graph.memlock"(%arg0) { node_id = 3, node_name = "add1", cluster_id = 1 } () -> ()
        "graph.memlock"(%arg1) { node_id = 3, node_name = "add1", cluster_id = 1 } () -> ()
        %res = "graph.add"(%arg0, %arg1) { tensor_addr = 7, node_id = 3, node_name = "add1", cluster_id = 1 } : (tensor<3xf32>, tensor<3xf32>) -> tensor<3xf32>
        "graph.memrelease"(%arg0) { node_id = 3, node_name = "add1", cluster_id = 1 } () -> ()
        "graph.memrelease"(%arg1) { node_id = 3, node_name = "add1", cluster_id = 1 } () -> ()
        "graph.return"() : () -> ()
    }
}
```

## Operations reference

There's an automatically generated [reference page](02_mlir_graph_reference.md) for Graph dialect.

# Lowering to LLVM IR

TBD