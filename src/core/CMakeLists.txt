set(ATH_CORE "AthenaCore" CACHE STRING "" FORCE)

file(GLOB_RECURSE CORE_SRC "*.cpp")

add_athena_library(${ATH_CORE} STATIC ${CORE_SRC})