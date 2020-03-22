include "builtins_base.td"

def rt_all_types = [float, double]

def RT : Runtime<rt_all_types> {
    let name = "CPU Runtime";
    let version = "0.1.0";
    let summary = "CPU Runtime";
    let description = "TBD";
}

include "../builtins.td"