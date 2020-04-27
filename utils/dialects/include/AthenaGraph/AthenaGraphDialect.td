#ifndef ATHENA_GRAPH_DIALECT
#define ATHENA_GRAPH_DIALECT

include "mlir/IR/OpBase.td"

def AthenaGraph_Dialect : Dialect {
  let name = "ath_graph";

  // The C++ namespace that the dialect class definition resides in.
  let cppNamespace = "ath_graph";
}

class AthenaGraph_Op<string mnemonic, list<OpTrait> traits = []> :
    Op<AthenaGraph_Dialect, mnemonic, traits>;

#endif // ATHENA_GRAPH_DIALECT
