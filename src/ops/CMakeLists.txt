set(ATH_OPS "AthenaOps" CACHE STRING "" FORCE)

add_athena_library(${ATH_OPS} OBJECT
        AddOperation.cpp
        MSELossFunction.cpp
        GEMMOperation.cpp)