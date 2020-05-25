add_mlir_dialect_library(MLIRAthenaGraph
        AthenaGraphDialect.cpp
        AthenaGraphOps.cpp
        ComputationalOpInterface.cpp

        ADDITIONAL_HEADER_DIRS
        ${PROJECT_SOURCE_DIR}/include
        ${PROJECT_BINARY_DIR}/include

        DEPENDS
        MLIRAthenaGraphOpsIncGen
        MLIRComputationalOpInterfaceIncGen
        )

target_link_libraries(MLIRAthenaGraph PUBLIC
                      MLIRIR MLIRCallInterfaces MLIRControlFlowInterfaces)
