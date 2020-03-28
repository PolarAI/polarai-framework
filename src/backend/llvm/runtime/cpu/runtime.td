include "runtime_base.td"

def RTAllTypes {
    list<Type> type = [float,double];
}
def RT : Runtime<RTAllTypes.type> {
    let name = "CPU Runtime";
    let version = "0.1.0";
    let summary = "CPU Runtime";
    let description = "TBD";
}

include "builtins_base.td"
