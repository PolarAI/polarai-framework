add_mlir_dialect_library(MLIRPolarGraph
        PolarGraphDialect.cpp
        PolarGraphOps.cpp
        ComputationalOpInterface.cpp
        PeepholePatterns.cpp
        kernels/AddOp.cpp
        kernels/CopyOp.cpp
        kernels/DivideOp.cpp
        kernels/MatMulOp.cpp
        kernels/MulConcatOp.cpp
        kernels/MulOp.cpp
        kernels/SigmoidOp.cpp
        kernels/ReLUOp.cpp
        kernels/Conv2DOp.cpp
        kernels/Pool2DOp.cpp
        kernels/SoftmaxOp.cpp

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
