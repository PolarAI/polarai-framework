#ifndef POLAR_GRAPH_DIALECT
#define POLAR_GRAPH_DIALECT

include "mlir/IR/OpBase.td"
include "ComputationalOpInterface.td"

def PolarGraph_Dialect : Dialect {
  let name = "polar_graph";

  // The C++ namespace that the dialect class definition resides in.
  let cppNamespace = "polar_graph";
}

class PolarGraph_Op<string mnemonic, list<OpTrait> traits = []> :
    Op<PolarGraph_Dialect, mnemonic, traits>;

#endif // POLAR_GRAPH_DIALECT
