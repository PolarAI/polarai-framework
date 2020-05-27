add_mlir_dialect_library(MLIRAthenaRuntime
        AthenaRuntimeDialect.cpp
        AthenaRuntimeOps.cpp

        ADDITIONAL_HEADER_DIRS
        ${PROJECT_SOURCE_DIR}/include
        ${PROJECT_BINARY_DIR}/include

        DEPENDS
        MLIRAthenaRuntimeOpsIncGen
        )

target_link_libraries(MLIRAthenaRuntime PUBLIC
     MLIRIR
     MLIRCallInterfaces
     MLIRControlFlowInterfaces)

