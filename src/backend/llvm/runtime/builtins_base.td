include "runtime_base.td"

// MARK: Utility operations
def Allocate : RT_Builtin<"allocate", [void]> {
    let summary = "memory allocation";
    let description = [{
Allocate memory for tensor.
    }];
}

def Fill : RT_Builtin<"fill", RTAllTypes.type> {
    let summary = "pattern fill";
    let description = [{
Fills tensor A with the value provided.
    }];
    let arguments = (ins tensor:$a, gentype:$pattern);
}

// MARK: Arithmetic operations

def Add : RT_Builtin<"add", RTAllTypes.type> {
    let summary = "element-wise addition";
    let description = [{
Performs element-wise addition and stores result in tensor C. The following formula describes the
operation:
```
C = alpha * A + beta * B + gamma * C
```
where `alpha`, `beta`, and `gamma` are scaling factors, `A`, `B`, and `C` are tensors of the same
size and data type.
    }];
    let arguments = (ins gentype:$alpha,
                         tensor:$a,
                         gentype:$beta,
                         tensor:$b,
                         gentype:$beta,
                         tensor:$c);
}

def EWMul : RT_Builtin<"mul", RTAllTypes.type> {
    let summary = "element-wise multiplication";
    let description = [{
Performs element-wise multiplication and stores result in tensor C. The following formula describes
the operation:
```
C = alpha * A * B + beta * C
```
where `alpha` and `beta` are scaling factors, `A`, `B`, and `C` are tensors of the same size and
data type.
    }];
    let arguments = (ins gentype:$alpha,
                         tensor:$a,
                         tensor:$b,
                         gentype:$beta,
                         tensor:$c);
}

// MARK: Linear algebra

def MatMul : RT_Builtin<"matmul", RTAllTypes.type> {
    let summary = "matrix-matrix multiplication";
    let description = [{
Performs matrix-matrix multiplication on rank 2 tensors and stores the result to tensor C. Formula:
```
C = alpha * A . B + beta * C;
```
where `alpha` and `beta` are scaling factors; `A`, `B`, and `C` are rank 2 tensors, tensor `A` has
shape `M x K`, tensor `B` has shape `K x N`, tensor `C` has shape `M x N`, and all tensors have the
same data type.
    }];
    let arguments = (ins gentype:$alpha,
                         tensor:$a,
                         tensor:$b,
                         gentype:$beta,
                         tensor:$c);
}

def Conv : RT_Builtin<"conv", RTAllTypes.type> {
    let summary = "tensor convolution operation";
    let description = [{TBD}];
    let arguments = (ins tensor:$inp, tensor:$kernel, tensor:$out);
}
