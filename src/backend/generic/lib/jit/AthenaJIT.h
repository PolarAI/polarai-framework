#include <llvm/ExecutionEngine/Orc/LLJIT.h>
#include <llvm/Support/FileSystem.h>
#include <mlir/IR/Module.h>
#include <mlir/Pass/PassManager.h>

namespace athena::backend::llvm {
class Device;
struct ProgramDesc;
class AthenaJIT {
public:
  AthenaJIT(std::unique_ptr<::llvm::orc::LLJIT> jit);

  static auto create() -> std::shared_ptr<AthenaJIT>;
  static auto createWithDebugging() -> std::shared_ptr<AthenaJIT>;

  void addModule(const mlir::OwningModuleRef& ref);
  auto lookupSymbol(::llvm::StringRef symbolName) -> ::llvm::JITTargetAddress;

  auto getContext() -> mlir::MLIRContext* { return &mContext; }

  void registerDevice(std::shared_ptr<Device>);
  void resetDevices();

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
  std::vector<std::shared_ptr<Device>> mRegisteredDevices;
  std::vector<std::shared_ptr<ProgramDesc>> mCompiledPrograms;
};
} // namespace athena::backend::llvm
