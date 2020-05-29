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
  setupMlirPassManager();
};
auto AthenaJIT::create() -> std::shared_ptr<AthenaJIT> {
  ::llvm::sys::DynamicLibrary::LoadLibraryPermanently(nullptr);
  auto JIT = ExitOnErr(LLJITBuilder().create());
  JIT->getMainJITDylib().addGenerator(
      ::llvm::cantFail(
          ::llvm::orc::DynamicLibrarySearchGenerator::GetForCurrentProcess(
              JIT->getDataLayout().getGlobalPrefix())));

  return std::make_shared<AthenaJIT>(std::move(JIT));
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
  mMlirPassManager.addPass(mlir::createCanonicalizerPass());
  mMlirPassManager.addPass(mlir::createGraphRelationDestructorPass());
  mMlirPassManager.addPass(mlir::createLowerGraphToRuntimePass());
  auto& funcOpt = mMlirPassManager.nest<mlir::FuncOp>();
  funcOpt.addPass(mlir::createBarrierLegalizerPass());
  funcOpt.addPass(mlir::createLegalizeRTForLoweringPass());
  mMlirPassManager.addPass(mlir::createDeployDefaultFunctionsPass());
  mMlirPassManager.addPass(mlir::createLowerRuntimeToLLVMPass());
}
void AthenaJIT::compileModule() {
  auto res = mMlirPassManager.run(*mInternalModule);
  if (mlir::failed(res)) {
    // todo throw a real error.
    ::llvm::errs() << "JIT error\n";
  }

  auto llvmModule = mlir::LLVM::ModuleTranslation::translateModule(
      mInternalModule->getOperation());

  std::unique_ptr<LLVMContext> llvmCtx = std::make_unique<LLVMContext>();
  auto newModule =
      mlir::LLVM::cloneModuleIntoNewContext(llvmCtx.get(), llvmModule.get());

  ThreadSafeModule tsm(std::move(newModule), std::move(llvmCtx));
  auto err = mJITInstance->addIRModule(std::move(tsm));
  if (err) {
    // todo throw a real error.
    llvm_unreachable("Unexpected error");
  }
}
} // namespace athena::backend::llvm
