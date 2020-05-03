#ifndef ATHENA_RUNTIME_DIALECT
#define ATHENA_RUNTIME_DIALECT

include "mlir/IR/OpBase.td"

def AthenaRuntime_Dialect : Dialect {
  let name = "ath_rt";
  let cppNamespace = "ath_rt";
}

class AthenaRuntime_Op<string mnemonic, list<OpTrait> traits = []> :
    Op<AthenaRuntime_Dialect, mnemonic, traits>;

#endif // ATHENA_RUNTIME_DIALECT
