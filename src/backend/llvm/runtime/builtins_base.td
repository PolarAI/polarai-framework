include "runtime_base.td"

def Allocate_Base : RT_Builtin<"allocate", [void]> {
    let summary = "memory allocation";
    let description = [{
Allocate memory for tensor.
    }];
}

class Add_Base<list<Type> gentype = []> : RT_Builtin<"add", gentype> {
    let summary = "element-wise addition";
    let description = [{
Add corresponding elements of one tensor to another. Tenors are expected to match.
    }];
    let arguments = (ins tensor:$a, tensor:$b);
}

class Hadamard_Base<list<Type> gentype = []> : RT_Builtin<"hadamard", gentype> {
    let summary = "element-wise multiplication";
    let description = [{
Multiply corresponding elements of one tensor to another. Tenors are expected to match.
    }];
    let arguments = (ins tensor:$a, tensor:$b);
}

class Gemm_Base<list<Type> gentype = []> : RT_Builtin<"hadamard", gentype> {
    let summary = "generic matrix-matrix multiplication";
    let description = [{
Perform matrix-matrix multiplication.
    }];
    let arguments = (ins tensor:$a, tensor:$b);
}

class Fill_Base<list<Type> gentype = []> : RT_Builtin<"hadamard", gentype> {
   let summary = "pattern fill";
   let description = [{
Fill tensor memory with pattern.
   }];
   let arguments = (ins tensor:$a);
}