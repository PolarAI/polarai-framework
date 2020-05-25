set(ATH_LOADERS "AthenaLoaders" CACHE STRING "" FORCE)

file(GLOB_RECURSE LOADERS_SRC "*.cpp")

add_athena_library(${ATH_LOADERS} OBJECT
        ATH_LOADERS # export name
        loaders/loaders_export.h # export file name
        ${LOADERS_SRC})
