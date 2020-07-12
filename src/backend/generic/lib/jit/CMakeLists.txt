set(PolarJitTarget "PolarJIT" CACHE STRING "" FORCE)

add_polar_library(${PolarJitTarget} OBJECT
        PolarJIT.cpp)

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
target_include_directories(${PolarJitTarget} PUBLIC 
                           ${PROJECT_BINARY_DIR}/utils/dialects/include
                           ${PROJECT_SOURCE_DIR}/utils/dialects/include)
target_include_directories(${PolarJitTarget} PUBLIC ${LLVM_INCLUDE_DIRS})
target_compile_definitions(${PolarJitTarget} PRIVATE ${LLVM_DEFINITIONS})

get_property(dialect_libs GLOBAL PROPERTY MLIR_DIALECT_LIBS)
get_property(conversion_libs GLOBAL PROPERTY MLIR_CONVERSION_LIBS)
target_link_libraries(${PolarJitTarget} PUBLIC
                      ${llvm_libs}
                      MLIRIR
                      MLIRPolarGraph
                      MLIRPolarRuntime
                      MLIRPolarConversion
                      MLIRPolarPasses
                      MLIRCompute
                      ${dialect_libs}
                      ${conversion_libs}
                      MLIRTargetNVVMIR
                      )                       
