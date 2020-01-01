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
mandatory attributes `node_name` and `cluster_id`. These can be omitted only for
`graph.return` operation.

Here's an example of a simple graph:

```mlir
module {
    func @add() {
        %arg0 = "graph.alloca"() { dims = 1, tensor_addr = 1, tensor_type = f32, tensor_shape = dense<[3]>, node_name = "inputA", cluster_id = 0} : () -> tensor<*xf32>
        "graph.call"(%arg0, 0x0001) { callee = @MemoryLoaderLoad, node_name = "inputA", cluster_id = 0} : () -> ()
        %arg1 = "graph.alloca"() { dims = 1, tensor_addr = 1, tensor_type = f32, tensor_shape = dense<[3]>, node_name = "inputA", cluster_id = 0} : () -> tensor<*xf32>
        "graph.call"(%arg1, 0x0020) { callee = @MemoryLoaderLoad, node_name = "inputA", cluster_id = 0} : () -> ()
        "graph.memlock"(%arg0) { node_name = "add1", cluster_id = 1 } () -> ()
        "graph.memlock"(%arg1) { node_name = "add1", cluster_id = 1 } () -> ()
        %res = "graph.add"(%arg0, %arg1) { node_name = "add1", cluster_id = 1 } : (tensor<*xf32>, tensor<*xf32>) -> tensor<*xf32>
        "graph.memrelease"(%arg0) { node_name = "add1", cluster_id = 1 } () -> ()
        "graph.memrelease"(%arg1) { node_name = "add1", cluster_id = 1 } () -> ()
        "graph.return"() : () -> ()
    }
}
```

## Operations reference

### Control flow operations

#### `graph.call (?) -> ?`

The `graph.call` operation allows one to make calls to arbitrary functions outside the graph.
This is useful for loader invocation and user-defined operations.

**Mandatory attributes**
- `callee` **Function** - Function to be called

#### `graph.return () -> ()`

The `graph.return` operation denotes the end of the graph.

### Arithmetic operations

#### `graph.add (tensor<axbxcxd>, tensor<axbxcxd>) -> tensor<axbxcxd>`

The `graph.add` operation performs element-wise addition of given arguments.

#### `graph.mul (tensor<axbxcxd>, tensor<axbxcxd>) -> tensor<axbxcxd>` 

The `graph.add` operation performs element-wise multiplication of given arguments.

#### `graph.matmul (tensor<axbxc>, tensor<axbxc>) -> tensor<axbxc>`

The `graph.matmul` operation performs matrix-matrix multiplication of given arguments.

`C = alpha * A * B + beta * C`

**Mandatory attributes**
- `transpose_a` **Bool** - Transpose matrix A
- `transpose_b` **Bool** - Transpose matrix B
- `alpha` **Float** - Alpha coefficient
- `beta` **Float** - Beta coefficient

### Memory operations

#### `graph.alloca () -> tensor<*>`

The `graph.alloca` operation performs memory allocation for tensor. 

**Mandatory attributes**
- `tensor_addr` **Integer** -- Virtual address of tensor
- `tensor_dims` **Integer** - Number of tensor shape dimensions
- `tensor_type` **Type** - Type of data in tensor
- `tensor_shape` **Array of integer** - Size of tensor for each dimension

#### `graph.memlock () -> ()`

The `graph.memlock` operation locks tensor in device memory preventing it from being
forced out by other memory operations. 

#### `graph.memrelease () -> ()`

The `graph.memrelease` operation marks memory region as free.

### Utility operations

#### `graph.reshape (tensor<axbxc>) -> tensor<dxexf>`

The `graph.reshape` operation changes the shape of a tensor without data movement.

# Lowering to LLVM IR

TBD
