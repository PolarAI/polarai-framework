#ifndef CONVERT_COMPUTE_TO_SPIRV
#define CONVERT_COMPUTE_TO_SPIRV

include "Compute/ComputeOps.td"
include "mlir/Dialect/SPIRV/SPIRVStructureOps.td"

def : Pat<(Compute_ModuleEndOp), (SPV_ModuleEndOp)>;

#endif // CONVERT_COMPUTE_TO_SPIRV
