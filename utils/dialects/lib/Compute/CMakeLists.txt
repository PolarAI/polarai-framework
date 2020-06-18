add_mlir_dialect_library(MLIRCompute
        ComputeDialect.cpp
        ComputeOps.cpp
        ADDITIONAL_HEADER_DIRS
        ${PROJECT_SOURCE_DIR}/dialects/include

        DEPENDS
        MLIRComputeOpsIncGen

        LINK_LIBS PUBLIC
        MLIREDSC
        MLIRIR
        MLIRSideEffectInterfaces
        MLIRStandardOps
        MLIRSupport
        MLIRTransformUtils
        )
