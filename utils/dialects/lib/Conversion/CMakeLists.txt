add_subdirectory(ComputeToSPIRV)

add_mlir_conversion_library(MLIRPolarConversion
        GraphToRuntimePass.cpp
        ArgInfo.cpp
        RuntimeToLLVM.cpp

        ADDITIONAL_HEADER_DIRS
        ${PROJECT_SOURCE_DIR}/include/
        ${PROJECT_BINARY_DIR}/include

        LINK_LIBS PUBLIC
        MLIRIR
        MLIRPass
        MLIRPolarGraph
        MLIRPolarRuntime
        MLIRTransforms
        MLIRLLVMIR
        MLIRLLVMIRTransforms
        MLIRStandardToLLVM
        MLIRTargetLLVMIR
        MLIRTargetLLVMIRModuleTranslation
)
