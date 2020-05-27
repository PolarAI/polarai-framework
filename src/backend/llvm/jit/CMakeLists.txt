set(AthenaJitTarget "AthenaJIT" CACHE STRING "" FORCE)

add_athena_library(${AthenaJitTarget} OBJECT
        DONT_CATE
        dontcare.h
        AthenaJIT.cpp)

message(STATUS ${PROJECT_BINARY_DIR}/utils/dialects/include)

llvm_map_components_to_libnames(llvm_libs
        core
        irreader
        support
        transformutils
        passes
        codegen
        orcjit
        AllTargetsCodeGens AllTargetsAsmParsers AllTargetsDescs AllTargetsInfos
        lto)
target_include_directories(${AthenaJitTarget} PUBLIC 
                           ${PROJECT_BINARY_DIR}/utils/dialects/include
                           ${PROJECT_SOURCE_DIR}/utils/dialects/include)
target_link_libraries(${AthenaJitTarget} PUBLIC
                      ${llvm_libs}
                      MLIRIR
                      MLIRAthenaGraph
                      MLIRAthenaRuntime
                      MLIRAthenaConversion
                      MLIRAthenaPasses)                       
