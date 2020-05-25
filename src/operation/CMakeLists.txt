set(ATH_OPERATION "AthenaOperation" CACHE STRING "" FORCE)

file(GLOB_RECURSE OPERATION_SRC "*.cpp")

add_athena_library(${ATH_OPERATION} OBJECT
        ATH_OPERATION # export name
        operation/operation_export.h # export file name
        ${OPERATION_SRC})
