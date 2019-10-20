---
id: llvm_builtins
title: LLVM Backend: Builtins
---

# LLVM Backend: Builtins

## What is a builtin?

A builtin is a routine available in runtime that you
can use to perform simple actions on tensors. Available
builtins include such operations as matrix-matrix multiplication,
Hadamard product, vector sum. You can find all standard
builtins and their signatures in `include/athena/backend/llvm/runtime`
directory.

## How to add builtin?

### Step 1. Add method signature

Go to `include/athena/backend/llvm/runtime` and create signature
for your new builtin. The first two arguments must be `Device*`
and `Allocator*`. Check out existing builtins for reference.

**Note.** If you create a new header file, make sure to include
it in builtin.h file.

### Step 2. Implement builtin

Go to `src/backend/llvm/runtime-cpu` and create implementation
for your new builtin.

### Step 3.  Generate glue code for builtin

There's a glue code generator that you can use to generate
correctly mangled C-style wrappers and LLVM IR bindings for them.

To make use of this generator, add your new builtin to
`src/backend/llvm/builtins.td`. It has the following syntax:

```
builtin-name : Arg1Type, Arg2Type, ... : SpecializationType1, SpecializationType2, ...
``` 

Where `builtin-name` is a name of your newly created builtin.
`ArgNType` is any valid C++ type or `gentype` keyword, that will
be replaced by a real typename. `SpecializationTypeN` is a list of
typenames to be used as template parameter variants for your builtin.
If not specified, the default list of `float` and `double` will be used.

### Step 4. Register your builtin

Register builtin generator in `src/backend/llvm/codegen/register_default_functors.cpp`.
Use one of predefined routines or create your own.