#ifndef CLANG_OPS
#define CLANG_OPS

include "mlir/IR/OpBase.td"

def Clang_Dialect : Dialect {
  let name = "clang";

  // The C++ namespace that the dialect class definition resides in.
  let cppNamespace = "clang";
}

def Clang_StructType :
    Type<CPred<"$_self.isa<StructType>()">, "Clang struct type">;

def Clang_RawPointerType :
    Type<CPred<"$_self.isa<RawPointerType>()">, "Clang pointer type">;

class Clang_Op<string mnemonic, list<OpTrait> traits = []> :
    Op<Clang_Dialect, mnemonic, traits>;

def GepOp : Clang_Op<"getelementprt", [NoSideEffect]> {
  let summary = "get element pointer operation";
  let description = [{
    Extracts pointer to an element of a structure.
  }];

  let arguments = (ins AnyType:$target, I64Attr:$idx);
  let results = (outs AnyType:$resPtr);
  let builders = [
    OpBuilder<"Builder* builder, OperationState& result, "
              "Value ptr, size_t index", [{
                result.addOperands(ptr);
                auto idx = builder->getI64IntegerAttr(index);
                result.addAttribute("idx", idx);
                result.types.push_back(ptr.getType()); // fixme
              }]>
    ];
}

def StructDeclOp : Clang_Op<"struct_decl", [NoSideEffect, HasParent<"ModuleOp">]> {
  let summary = "structure declaration operation";
  let description = [{ tbd }];
  let arguments = (ins StrAttr:$structName, TypeAttr:$structType);
  let builders = [
    OpBuilder<"Builder* builder, OperationState& result, "
              "llvm::StringRef name, StructType type">
    ];
}

def LoadOp : Clang_Op<"load"> {
  let summary = "load operation";
  let description = [{ tbd }];
  let arguments = (ins Clang_RawPointerType:$ptr);
  let results = (outs AnyType);
}

def StoreOp : Clang_Op<"store"> {
  let summary = "store operation";
  let description = [{ tbd }];
  let arguments = (ins Clang_RawPointerType:$ptr, AnyType:$value);
}

def AllocaOp : Clang_Op<"alloca"> {
  let summary = "allocation operation";
  let description = [{ tbd }];
  let arguments = (ins Index:$size, TypeAttr:$resType);
  let results = (outs Clang_RawPointerType);

  let builders = [OpBuilder<
    "Builder *builder, OperationState &result, "
    "Value size, RawPointerType ptrType", [{
      result.addOperands(size);
      result.types.push_back(ptrType);
    }]>];
}

#endif // CLANG_OPS