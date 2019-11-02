set(ATH_MEM_LOADER "AthenaMemoryLoader" CACHE STRING "" FORCE)
add_athena_library(${ATH_MEM_LOADER} OBJECT MemoryLoader.cpp)
target_link_libraries(${ATH_MEM_LOADER} PRIVATE ${ATH_CORE})