#ifndef ATHENA_RUNTIME_DIALECT
#define ATHENA_RUNTIME_DIALECT

include "mlir/IR/OpBase.td"

def AthenaRuntime_Dialect : Dialect {
  let name = "ath_rt";
  let cppNamespace = "ath_rt"; 

  let extraClassDeclaration = [{
    mlir::Type parseType(mlir::DialectAsmParser& parser) const override;
    void printType(mlir::Type type, mlir::DialectAsmPrinter& printer) const override;
  }];  
}

class AthenaRuntime_Op<string mnemonic, list<OpTrait> traits = []> :
    Op<AthenaRuntime_Dialect, mnemonic, traits>;

def AthenaRuntime_DeviceType : Type<CPred<"$_self.isa<DeviceType>()">,
                                          "Athena Runtime Device type">;
def AthenaRuntime_EventType : Type<CPred<"$_self.isa<EventType>()">,
                                         "Athena Runtime Event type">;
def AthenaRuntime_GraphHandleType : Type<CPred<"$_self.isa<GraphHandleType>()">,
                                             "Athena Runtime GraphHandle type">;

#endif // ATHENA_RUNTIME_DIALECT