add_mlir_dialect_library(MLIRPolarRuntime
        PolarRuntimeDialect.cpp
        PolarRuntimeOps.cpp

        ADDITIONAL_HEADER_DIRS
        ${PROJECT_SOURCE_DIR}/include
        ${PROJECT_BINARY_DIR}/include

        DEPENDS
        MLIRPolarRuntimeOpsIncGen
        )

target_link_libraries(MLIRPolarRuntime PUBLIC
     MLIRIR
     MLIRCallInterfaces
     MLIRSideEffectInterfaces
     MLIRControlFlowInterfaces)

