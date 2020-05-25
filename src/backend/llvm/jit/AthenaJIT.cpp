#include "AthenaJIT.h"

#include "AthenaGraph/AthenaGraphDialect.h"
#include "AthenaRuntime/AthenaRuntimeDialect.h"
#include "Conversion/GraphToRuntimePass.h"
#include "Conversion/RuntimeToLLVM.h"
#include "Passes/Passes.h"

#include "mlir/IR/Builders.h"
#include "mlir/IR/Dialect.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/InitAllDialects.h"
#include "mlir/InitAllPasses.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Pass/PassManager.h"
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

  mlir::registerDialect<mlir::ath_graph::AthenaGraphDialect>();
  mlir::registerDialect<mlir::ath_rt::AthenaRuntimeDialect>();
  setupMlirPassManager();
};
auto AthenaJIT::create() -> AthenaJIT {
  InitializeNativeTarget();
  InitializeNativeTargetAsmPrinter();

  auto JIT = ExitOnErr(LLJITBuilder().create());

  return AthenaJIT(std::move(JIT));
}

void AthenaJIT::addModule(mlir::OwningModuleRef ref) {
  mlir::OpBuilder builder(&mContext);
  if (!mInternalModule) {
    mInternalModule = mlir::OwningModuleRef(
        builder.create<mlir::ModuleOp>(builder.getUnknownLoc()));
  }

  builder.setInsertionPointToStart(mInternalModule->getBody());

  for (auto& op : *ref) {
    builder.clone(op);
  }
}
auto AthenaJIT::lookupSymbol(::llvm::StringRef symbolName)
    -> ::llvm::JITTargetAddress {
  if (mInternalModule) {
    compileModule();
  }

  return ExitOnErr(mJITInstance->lookup(symbolName)).getAddress();
}
void AthenaJIT::setupMlirPassManager() {
  mlir::OpPassManager& modulePassManager =
      mMlirPassManager.nest<mlir::ModuleOp>();
  modulePassManager.addPass(mlir::createCanonicalizerPass());
  modulePassManager.addPass(mlir::createGraphRelationDestructorPass());
  modulePassManager.addPass(mlir::createLowerGraphToRuntimePass());
  modulePassManager.addPass(mlir::createBarrierLegalizerPass());
  modulePassManager.addPass(mlir::createLowerRuntimeToLLVMPass());
}
void AthenaJIT::compileModule() {
  auto res = mMlirPassManager.run(*mInternalModule);
  // todo check result

  auto& llvmModule =
      mContext.getRegisteredDialect<mlir::LLVM::LLVMDialect>()->getLLVMModule();

  ThreadSafeModule tsm(llvmModule);
  mJITInstance->addIRModule(llvmModule);
}
} // namespace athena::backend::llvm
