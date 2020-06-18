add_subdirectory(ComputeToSPIRV)

add_mlir_conversion_library(MLIRAthenaConversion
        GraphToRuntimePass.cpp
        ArgInfo.cpp
        RuntimeToLLVM.cpp

        ADDITIONAL_HEADER_DIRS
        ${PROJECT_SOURCE_DIR}/include/
        ${PROJECT_BINARY_DIR}/include

        DEPENDS
        MLIRAthenaGraph

        LINK_LIBS PUBLIC
        MLIRIR
        MLIRPass
        MLIRAthenaGraph
        MLIRAthenaRuntime
        MLIRTransforms
        MLIRLLVMIR
        MLIRLLVMIRTransforms
        MLIRStandardToLLVM
        MLIRTargetLLVMIR
        MLIRTargetLLVMIRModuleTranslation
)
