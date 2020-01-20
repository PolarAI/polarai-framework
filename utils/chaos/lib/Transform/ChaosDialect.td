#ifndef CHAOS_OPS
#define CHAOS_OPS

include "mlir/IR/OpBase.td"

def Chaos_Dialect : Dialect {
  let name = "chaos";
  let cppNamespace = "chaos";
}

class Chaos_Op<string mnemonic, list<OpTrait> traits = []> :
                  Op<Chaos_Dialect, mnemonic, traits>;

def ReinterpretOp : Chaos_Op<"reinterpret", [NoSideEffect]> {
  let summary = "reinterpret cast operation";
  let description = [{}];
  let arguments = (ins AnyMemRef:$source);
  let results = (outs AnyMemRef:$out);

  let builders = [OpBuilder<"Builder *builder, OperationState &result, "
                            "Value source, Type destType">];
}

#endif // CHAOS_OPS