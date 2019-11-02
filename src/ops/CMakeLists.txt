set(ATH_OPS "AthenaOps" CACHE STRING "" FORCE)

add_athena_library(${ATH_OPS} OBJECT
        ATH_OPS # export name
        ops/ops_export.h # export file name
        AddOperation.cpp
        MSELossFunction.cpp
        GEMMOperation.cpp)