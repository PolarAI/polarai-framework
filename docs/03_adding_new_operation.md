---
id: add_ops
title: Adding Custom Operations
---

# Adding Custom Operation

## Step 1. Create new Operation class

Create a class derived from ahtena::core::Operation.
For this class the following methods must be implemented:

- `core::inner::Tensor *getResultTensor(std::vector<core::inner::Tensor *> args) const`: Generates tensor of proper size and type for operation result based on incoming arguments.
- `core::inner::Tensor *getDerivativeTensor(std::vector<core::inner::Tensor *> args, int argNo) const`: Generates tensor of proper size and type for operation derivative result based on incoming arguments.
- `void gen(core::AbstractGenerator &g, std::vector<core::inner::Tensor *> &operationArguments) const`: Generates code for operation using provided Generator
- `void genDerivative(int order,
                          core::AbstractGenerator &g,
                          core::inner::Tensor &operationResult,
                          std::vector<core::inner::Tensor *> &operationArguments,
                          core::inner::Tensor &derivativeTensor,
                          int argNo) const override;`: Generates code for operation derivative using provided Generator.
- `size_t getOperandsCount() const`: Returns count of required operands.

If you want to add your operation to set of default, class must 
reside in `athena::ops` namespace. 

## Step 2. Implement code generation

Operations use Generator interface to build program code
for Graph evaluation. 

Generator defines method `gen` that can be used to generate
a call to builtin. To learn more about builtins, check out
LLVM backend guide.