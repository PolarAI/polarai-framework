set(ATH_UTILS "AthenaUtils" CACHE STRING "" FORCE)

file(GLOB_RECURSE UTILS_SRC "*.cpp")

add_athena_library(${ATH_UTILS} STATIC
        ATH_UTILS # export name
        utils/utils_export.h # export file name
        ${UTILS_SRC})
