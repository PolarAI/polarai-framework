add_mlir_dialect_library(MLIRPolarGraph
        PolarGraphDialect.cpp
        PolarGraphOps.cpp
        ComputationalOpInterface.cpp
        PeepholePatterns.cpp

        ADDITIONAL_HEADER_DIRS
        ${PROJECT_SOURCE_DIR}/include
        ${PROJECT_BINARY_DIR}/include

        DEPENDS
        MLIRPolarGraphOpsIncGen
        MLIRComputationalOpInterfaceIncGen
        )

target_link_libraries(MLIRPolarGraph PUBLIC
                      MLIRIR
                      MLIRCallInterfaces 
                      MLIRSideEffectInterfaces
                      MLIRControlFlowInterfaces)
