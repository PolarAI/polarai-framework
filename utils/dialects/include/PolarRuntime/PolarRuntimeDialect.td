#ifndef POLAR_RUNTIME_DIALECT
#define POLAR_RUNTIME_DIALECT

include "mlir/IR/OpBase.td"

def PolarRuntime_Dialect : Dialect {
  let name = "polar_rt";
  let cppNamespace = "polar_rt"; 

  let extraClassDeclaration = [{
    mlir::Type parseType(mlir::DialectAsmParser& parser) const override;
    void printType(mlir::Type type, mlir::DialectAsmPrinter& printer) const override;
  }];  
}

class PolarRuntime_Op<string mnemonic, list<OpTrait> traits = []> :
    Op<PolarRuntime_Dialect, mnemonic, traits>;

def PolarRuntime_DeviceType : Type<CPred<"$_self.isa<DeviceType>()">,
                                          "Polar Runtime Device type">;
def PolarRuntime_EventType : Type<CPred<"$_self.isa<EventType>()">,
                                         "Polar Runtime Event type">;
def PolarRuntime_GraphHandleType : Type<CPred<"$_self.isa<GraphHandleType>()">,
                                             "Polar Runtime GraphHandle type">;

#endif // POLAR_RUNTIME_DIALECT
