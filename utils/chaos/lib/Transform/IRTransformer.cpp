//===----------------------------------------------------------------------===//
// Copyright (c) 2020 Athena. All rights reserved.
// https://getathena.ml
//
// Licensed under MIT license.
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS, WITHOUT
// WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the
// License for the specific language governing permissions and limitations under
// the License.
//===----------------------------------------------------------------------===//

#include "ChaosLoweringPass.h"
#include "IRCleanUpPass.h"
#include "LLVMToStdPass.h"
#include "LoopAffinitizer.h"
#include <Transform/IRTransformer.h>
#include <fstream>
#include <llvm/Bitcode/BitcodeWriter.h>
#include <llvm/IRReader/IRReader.h>
#include <llvm/Passes/PassBuilder.h>
#include <llvm/Support/SourceMgr.h>
#include <llvm/Transforms/Scalar/LoopSimplifyCFG.h>
#include <mlir/Conversion/StandardToLLVM/ConvertStandardToLLVMPass.h>
#include <mlir/Pass/Pass.h>
#include <mlir/Pass/PassManager.h>
#include <mlir/Target/LLVMIR.h>
#include <mlir/Transforms/Passes.h>

using namespace llvm;

namespace chaos {

std::unique_ptr<IRTransformer>
IRTransformer::getFromIrFile(const std::string& filename) {
  auto ctx = std::make_unique<llvm::LLVMContext>();
  SMDiagnostic err;
  auto module = llvm::parseIRFile(filename, err, *ctx, false, "");
  auto transformer =
      std::make_unique<IRTransformer>(std::move(module), std::move(ctx));

  return std::move(transformer);
}

IRTransformer::IRTransformer(std::unique_ptr<llvm::Module> llvmModule,
                             std::unique_ptr<llvm::LLVMContext> ctx)
    : mLLVMContext(std::move(ctx)), mDataLayout(llvmModule.get()) {
  // Set up LLVM optimizer infra
  mPassBuilder.registerModuleAnalyses(mModuleAnalysisManager);
  mPassBuilder.registerCGSCCAnalyses(mCGSCCAnalysisManager);
  mPassBuilder.registerFunctionAnalyses(mFunctionAnalysisManager);
  mPassBuilder.registerLoopAnalyses(mLoopAnalysisManager);
  mPassBuilder.crossRegisterProxies(
      mLoopAnalysisManager, mFunctionAnalysisManager, mCGSCCAnalysisManager,
      mModuleAnalysisManager);

  // Simplify module beforehand
  auto moduleSimplifier = mPassBuilder.buildModuleSimplificationPipeline(
      PassBuilder::O1, PassBuilder::ThinLTOPhase::None);
  moduleSimplifier.addPass(
      createModuleToFunctionPassAdaptor(LoopSimplifyPass()));
  moduleSimplifier.addPass(createModuleToFunctionPassAdaptor(LCSSAPass()));
  moduleSimplifier.run(*llvmModule, mModuleAnalysisManager);

  // Convert LLVMI IR to MLIR
  mMLIRModule = std::make_unique<mlir::OwningModuleRef>(
      mlir::translateLLVMIRToModule(std::move(llvmModule), &mMLIRContext));
}
void IRTransformer::dumpMLIR(const std::string& filename) {
  std::error_code err;
  llvm::raw_fd_ostream out(filename, err);
  mlir::OpPrintingFlags flags;
  flags.enableDebugInfo(true);
  mMLIRModule->get().print(out, flags);
  out.close();
}
void IRTransformer::run() {
  mlir::PassManager mlirPM(&mMLIRContext);
  mlirPM.disableMultithreading();
  mlirPM.enableCrashReproducerGeneration("crash");
  mlirPM.addPass(chaos::createLLVMToStdPass(&mMLIRContext));
  mlirPM.addPass(mlir::createCanonicalizerPass());
  mlirPM.addPass(chaos::createLoopAffinitizerPass());
  //  mlirPM.addPass(mlir::createCanonicalizerPass());
  mlirPM.addPass(chaos::createIRCleanUpPass());
  //    funcPM.addPass(chaos::createLoopAffinitizerPass());
  //  mlir::OpPassManager& optPM = mlirPM.nest<mlir::FuncOp>();
  //  optPM.addNestedPass<mlir::LLVM::LLVMFuncOp>(chaos::createLoopAffinitizerPass());
  //  optPM.addPass(mlir::createLoopFusionPass());
  //    mlirPM.addPass(chaos::createChaosLoweringPass(&mMLIRContext));
  //  mlirPM.addPass(mlir::createLowerToLLVMPass());
  if (mlir::failed(mlirPM.run(mMLIRModule->get()))) {
    dumpMLIR("test.mlir");
    llvm_unreachable("Err");
  }
  //  mMLIRModule->get().verify();
  dumpMLIR("test3.mlir");
  mLLVMModule = mlir::translateModuleToLLVMIR(mMLIRModule->get());
  dumpLLVMIR("test3.ll");

  auto moduleOptimizationPassManager =
      mPassBuilder.buildModuleOptimizationPipeline(::llvm::PassBuilder::O2,
                                                   false);
  auto functionSimplificationPassManager =
      mPassBuilder.buildFunctionSimplificationPipeline(
          ::llvm::PassBuilder::O2, ::llvm::PassBuilder::ThinLTOPhase::None,
          false);
  auto thinLTOPreLinkPassManager =
      mPassBuilder.buildThinLTOPreLinkDefaultPipeline(::llvm::PassBuilder::O2);

  moduleOptimizationPassManager.run(*mLLVMModule, mModuleAnalysisManager);
  for (auto& func : *mLLVMModule) {
    if (!func.isDeclaration())
      functionSimplificationPassManager.run(func, mFunctionAnalysisManager);
  }
  thinLTOPreLinkPassManager.run(*mLLVMModule, mModuleAnalysisManager);
}
void IRTransformer::dumpLLVMIR(const std::string& filename) {
  std::error_code err;
  llvm::raw_fd_ostream out(filename, err);
  mLLVMModule->print(out, nullptr);
  out.close();
}
void IRTransformer::writeBitcode(const std::string& filename) {
  std::error_code err;
  llvm::raw_fd_ostream out(filename, err);
  mLLVMModule->setDataLayout(mDataLayout);
  mLLVMModule->setTargetTriple(::llvm::sys::getDefaultTargetTriple());
  llvm::WriteBitcodeToFile(*mLLVMModule, out);
  out.close();
}
} // namespace chaos