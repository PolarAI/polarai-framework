#include <llvm/ExecutionEngine/Orc/LLJIT.h>
#include <mlir/IR/Module.h>
#include <mlir/Pass/PassManager.h>
#include <llvm/Support/FileSystem.h>

namespace athena::backend::llvm {
class AthenaJIT {
public:
  AthenaJIT(std::unique_ptr<::llvm::orc::LLJIT> jit);

  static auto create() -> std::shared_ptr<AthenaJIT>;
  static auto createWithDebugging() -> std::shared_ptr<AthenaJIT>;

  void addModule(const mlir::OwningModuleRef& ref);
  auto lookupSymbol(::llvm::StringRef symbolName) -> ::llvm::JITTargetAddress;

  auto getContext() -> mlir::MLIRContext* { return &mContext; }

private:
  void setupMlirPassManager();
  void compileModule();

  mlir::MLIRContext mContext;
  mlir::PassManager mMlirPassManager;
  mlir::OwningModuleRef mInternalModule;
  std::unique_ptr<::llvm::orc::LLJIT> mJITInstance;
#ifdef DEBUG
  ::llvm::SmallVector<char, 128> mTempFileGraph;
#endif
};
} // namespace athena::backend::llvm
