include "builtins_base.td"

def RT : Runtime<[double,float]> {
    let name = "CPU Runtime";
    let version = "0.1.0";
    let summary = "CPU Runtime";
    let description = "TBD";
}

def Add_Builtin : Add_Base<[float, double]>;
def Hadamard_Builtin : Hadamard_Base<[float, double]>;
def Gemm_Builtin : Gemm_Base<[float, double]>;
def Fill_Builtin : Fill_Base<[float, double]>;