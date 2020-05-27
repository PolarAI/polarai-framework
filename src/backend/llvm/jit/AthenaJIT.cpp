#include "AthenaJIT.h"

#include "AthenaGraph/AthenaGraphDialect.h"
#include "AthenaRuntime/AthenaRuntimeDialect.h"
#include "Conversion/GraphToRuntimePass.h"
#include "Conversion/RuntimeToLLVM.h"
#include "Passes/Passes.h"

#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/Dialect.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/InitAllDialects.h"
#include "mlir/InitAllPasses.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Target/LLVMIR/ModuleTranslation.h"
#include "llvm/ExecutionEngine/Orc/LLJIT.h"
#include "llvm/Support/InitLLVM.h"
#include "llvm/Support/TargetSelect.h"
#include "llvm/Support/raw_ostream.h"

using namespace ::llvm;
using namespace ::llvm::orc;

ExitOnError ExitOnErr;

namespace athena::backend::llvm {
AthenaJIT::AthenaJIT(std::unique_ptr<::llvm::orc::LLJIT> jit)
    : mJITInstance(std::move(jit)), mMlirPassManager(&mContext) {
  mlir::registerAllDialects();
  mlir::registerAllPasses();
  // mlir::registerAllTranslations();
  setupMlirPassManager();
};
auto AthenaJIT::create() -> std::unique_ptr<AthenaJIT> {
  auto JIT = ExitOnErr(LLJITBuilder().create());

  return std::make_unique<AthenaJIT>(std::move(JIT));
}

void AthenaJIT::addModule(const mlir::OwningModuleRef& ref) {
  mlir::OpBuilder builder(&mContext);
  if (!mInternalModule) {
    mInternalModule = mlir::OwningModuleRef(
        builder.create<mlir::ModuleOp>(builder.getUnknownLoc()));
  }

  builder.setInsertionPointToStart(mInternalModule->getBody());

  for (auto& op : *ref) {
    if (!::llvm::isa<mlir::ModuleTerminatorOp>(op)) {
      ::llvm::dbgs() << "Boom\n";
      builder.clone(op);
    }
  }
}
auto AthenaJIT::lookupSymbol(::llvm::StringRef symbolName)
    -> ::llvm::JITTargetAddress {
  if (mInternalModule) {
    compileModule();
    mInternalModule = nullptr;
  }

  return ExitOnErr(mJITInstance->lookupLinkerMangled(symbolName)).getAddress();
}
void AthenaJIT::setupMlirPassManager() {
  auto IRPrintingConfig =
      std::make_unique<mlir::PassManager::IRPrinterConfig>(true);
  mContext.disableMultithreading();
  mMlirPassManager.addPass(mlir::createGraphRelationDestructorPass());
  mMlirPassManager.addNestedPass<mlir::ModuleOp>(mlir::createLowerGraphToRuntimePass());
  // mMlirPassManager.addNestedPass<mlir::FuncOp>(mlir::createBarrierLegalizerPass());
  // mMlirPassManager.addPass(mlir::createLowerRuntimeToLLVMPass());
  mMlirPassManager.enableIRPrinting(std::move(IRPrintingConfig));
  mMlirPassManager.enableStatistics(mlir::PassDisplayMode::List);
}
void AthenaJIT::compileModule() {
  mContext.printOpOnDiagnostic(true);
  ::llvm::outs() << "Registered Dialects:\n";
  for (auto* dialect : mContext.getRegisteredDialects()) {
    ::llvm::outs() << dialect->getNamespace() << "\n";
  }
  for (auto &pass : mMlirPassManager.getPasses()) {
    ::llvm::outs() << pass.getName() << "\n";
  }
  auto res = mMlirPassManager.run(*mInternalModule);
  if (mlir::failed(res)) {
    ::llvm::errs() << "JIT error\n";
  }

  mInternalModule->dump();
  mInternalModule->print(::llvm::dbgs());
  // todo check result

  // auto llvmModule = mlir::LLVM::ModuleTranslation::translateModule(
  //     mInternalModule->getOperation());
  // llvmModule->print(::llvm::dbgs(), nullptr);

  // std::unique_ptr<LLVMContext> llvmCtx = std::make_unique<LLVMContext>();
  // auto newModule =
  //     mlir::LLVM::cloneModuleIntoNewContext(llvmCtx.get(), llvmModule.get());
  // newModule->print(::llvm::dbgs(), nullptr);

  // ThreadSafeModule tsm(std::move(newModule), std::move(llvmCtx));
  // auto err = mJITInstance->addIRModule(std::move(tsm));
  // if (err) {
  //   llvm_unreachable("Unexpected error");
  // }
}
} // namespace athena::backend::llvm
