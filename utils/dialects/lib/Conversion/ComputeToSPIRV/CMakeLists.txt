set(LLVM_TARGET_DEFINITIONS ComputeToSPIRV.td)
mlir_tablegen(ComputeToSPIRV.cpp.inc -gen-rewriters)
add_public_tablegen_target(MLIRComputeToSPIRVIncGen)
include_directories(
        ${PROJECT_BINARY_DIR}/lib/Conversion/ComputeToSPIRV/
)
add_mlir_conversion_library(MLIRComputeToSPIRV
        ComputeToOpenCLSPIRV.cpp

        ADDITIONAL_HEADER_DIRS
        ${PROJECT_SOURCE_DIR}/include/
        ${PROJECT_BINARY_DIR}/lib/Conversion/ComputeToSPIRV/

        DEPENDS
        MLIRCompute
        MLIRComputeToSPIRVIncGen
        )

target_include_directories(MLIRComputeToSPIRV PUBLIC ${PROJECT_BINARY_DIR}/lib/Conversion/ComputeToSPIRV/)
message(STATUS ${PROJECT_BINARY_DIR}/lib/Conversion/ComputeToSPIRV/)

target_link_libraries(MLIRComputeToSPIRV
        PUBLIC
        MLIRIR
        MLIRPass
        MLIRCompute
        MLIRTransforms
        MLIRSPIRV
        MLIRGPU
        MLIRSupport
        MLIRStandardToSPIRVTransforms
        MLIRStandardOps
        LLVMSupport
        MLIRSupport
        )
