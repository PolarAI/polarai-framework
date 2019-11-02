set(ATH_MEM_LOADER "AthenaMemoryLoader" CACHE STRING "" FORCE)
add_athena_library(${ATH_MEM_LOADER} OBJECT
        ATH_MEM_LOADER # export name
        loaders/MemoryLoader/mem_loader_export.h # export file name
        MemoryLoader.cpp)
target_link_libraries(${ATH_MEM_LOADER} PRIVATE ${ATH_CORE})