#ifndef CLANG_OPS
#define CLANG_OPS

include "mlir/IR/OpBase.td"

def Clang_Dialect : Dialect {
  let name = "clang";

  // The C++ namespace that the dialect class definition resides in.
  let cppNamespace = "clang";
}

class Clang_Op<string mnemonic, list<OpTrait> traits = []> :
    Op<Clang_Dialect, mnemonic, traits>;

def ForOp : Clang_Op<"for"> {
  let summary = "for operation";
  let description = [{
    Clang's `for` loop. Loop operation has 4 regions: init, condition,
    increment, and loop body.
  }];

  let regions = (region SizedRegion<1>:$init,
                        SizedRegion<1>:$cond,
                        SizedRegion<1>:$inc,
                        SizedRegion<1>:$body);

  let skipDefaultBuilders = 1;
  let builders = [
    OpBuilder<"Builder *builder, OperationState &result">
  ];

  let extraClassDeclaration = [{
    Block *getInit() { return &init().front(); }
    Block *getCond() { return &cond().front(); }
    Block *getInc() { return &inc().front(); }
    Block *getBody() { return &body().front(); }
  }];
}

#endif // CLANG_OPS