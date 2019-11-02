set(ATH_CORE "AthenaCore" CACHE STRING "" FORCE)

file(GLOB_RECURSE CORE_SRC "*.cpp")

add_athena_library(${ATH_CORE} STATIC
        ATH_CORE # export name
        core/core_export.h # export file name
        ${CORE_SRC})