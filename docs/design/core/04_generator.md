---
id: core_generator
title: "Low-level Code Generator"
---

# Low-level Code Generator

## Generation model

Athena Generator is a stateful object that forwards calls from GraphCompiler
to routines, brovided by the backend. An instance of the Generator stores
current generation point.

The native backend generator state can be anything that is required to build
an actual piece of code for the graph. For example, LLVM IR Module.

A pair of Graph name and Node ID uniquely defines position of code that is
about to be generated. However, to help backends generate parallel code,
a cluster ID is passed to backend routine. Unlike graph name or node ID, it
has no representation in the context. The cluster ID is an integer number from 0
to number of clusters - 1. It is safe to assume that cluster with ID `x - 1`
is fully executed before cluster with ID `x`.

## Backend functors

A backend functor is a routine that generates a piece of code for the Graph.
The backend must provide routines for all the builtins that are used across
the Graph. Otherwise a fatal error is raised.

> See also:
> 1. [List of mandatory builtins](0z_builtins.md)
> 2. [Builtins in LLVM backend](../llvm/04_builtins.md) 

All functors must meet the following requirements:

1. Be a function object with the following signature:
   ```cpp
   void functor(
        athena::core::Context &, // Athena Context
        std::string_view, // Graph name
        size_t, // Id of node to generate for
        size_t, // Id of cluster where node belongs to
        const std::vector<inner::Tensor> &, // Arguments to the builtin
        const std::any & // Builtin-specific options
        );
   ```
2. Be self-contained. A functor must capture everything it needs before being
   registered for the generator. This means that the functor may not be a pure
   function, and it can modify global context.
3. Thread safety is not required.

There's also one special functor for generating loads that must be provided by 
the backend:
```cpp
    void load_functor(
        athena::core::Context&, // Athena Context
        std::string_view, // Graph name
        size_t, // Id of node to generate for
        size_t, // Id of cluster where node belongs to
        inner::Tensor&, // Target tensor
        AbstractLoader* // Loader
    );
```

A user application can register its own functor for a specific builtin. However,
it is users obligation to ensure that their functor is compatible with the
underlying backend.

Operations can call Generator methods to check for builtins support and alter
behavior at runtime.