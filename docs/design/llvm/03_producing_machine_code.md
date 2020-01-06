---
id: producing_machine_code
title: Producing machine code
---

# Producing machine code

## LLVM IR in Athena

Athena provides users with [LLVM](https://llvm.org) backend, that uses framework's capabilities to produce executable
machine code for a graph. [LLVM IR](https://llvm.org/docs/LangRef.html) based representation of source code lies in the 
heart of the backend. It is used to represent both graph and all the utility routines that are needed to evaluate and
differentiate the expression.

### Basic layout of Graph program

Every graph is represented with a function that contains calls to functions, representing nodes. Every node is also a
function. It consists of a bunch of calls to allocator, builtins and utility functions. Node functions come with
metadata that indicated node ID, node name and cluster ID. This information is used to instrument graph code.

#### void* getDeviceForNode(size_t nodeId)

This is a special utility function that returns a pointer to a device that this node is dispatched to. This function
can come from another module to handle custom node dispatching. For JIT compilation this function will be evaluated
in compile time and inlined.

#### void* getAllocator()

This is a special utility function that provides a pointer to an Allocator. For JIT compilation this function will be 
evaluated in compile time and inlined.

#### void* getTensorPtr(size_t tensorAddr)

This is a special utility function that returns a pointer to a tensor by its virtual address.

### Key optimizations

TBD

## Just-in-time compiler

TBD

## Ahead of time compiler

TBD
