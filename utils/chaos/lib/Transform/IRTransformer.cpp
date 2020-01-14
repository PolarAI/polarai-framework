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

#include <Transform/IRTransformer.h>
#include <fstream>
#include <llvm/Analysis/CGSCCPassManager.h>
#include <llvm/Bitcode/BitcodeWriter.h>
#include <llvm/IR/PassManager.h>
#include <llvm/IRReader/IRReader.h>
#include <llvm/Passes/PassBuilder.h>
#include <llvm/Support/SourceMgr.h>
#include <mlir/Pass/Pass.h>
#include <mlir/Pass/PassManager.h>
#include <mlir/Target/LLVMIR.h>
#include <mlir/Transforms/Passes.h>

using namespace llvm;

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
  mMLIRModule = std::make_unique<mlir::OwningModuleRef>(
      mlir::translateLLVMIRToModule(std::move(llvmModule), &mMLIRContext));
}
void IRTransformer::dumpMLIR(const std::string& filename) {
  std::error_code err;
  llvm::raw_fd_ostream out(filename, err);
  mMLIRModule->get().print(out);
  out.close();
}
void IRTransformer::run() {
  mlir::PassManager mlirPM(&mMLIRContext);
  mlir::OpPassManager& optPM = mlirPM.nest<mlir::FuncOp>();
  optPM.addPass(mlir::createCanonicalizerPass());
  optPM.addPass(mlir::createLoopFusionPass());
  if (mlir::failed(mlirPM.run(mMLIRModule->get()))) {
    llvm_unreachable("Err");
  }

  mLLVMModule = mlir::translateModuleToLLVMIR(mMLIRModule->get());

  ::llvm::FunctionPassManager mFunctionSimplificationPassManager;
  ::llvm::ModulePassManager mModuleOptimizationPassManager;
  ::llvm::ModulePassManager mThinLTOPreLinkPassManager;

  LoopAnalysisManager loopAnalysisManager;
  ::llvm::FunctionAnalysisManager functionAnalysisManager;
  ::llvm::CGSCCAnalysisManager cGSCCAnalysisManager;
  ::llvm::ModuleAnalysisManager moduleAnalysisManager;

  ::llvm::PassBuilder passBuilder;
  passBuilder.registerModuleAnalyses(moduleAnalysisManager);
  passBuilder.registerCGSCCAnalyses(cGSCCAnalysisManager);
  passBuilder.registerFunctionAnalyses(functionAnalysisManager);
  passBuilder.registerLoopAnalyses(loopAnalysisManager);
  passBuilder.crossRegisterProxies(loopAnalysisManager, functionAnalysisManager,
                                   cGSCCAnalysisManager, moduleAnalysisManager);
  mModuleOptimizationPassManager = passBuilder.buildModuleOptimizationPipeline(
      ::llvm::PassBuilder::O1, false);
  mFunctionSimplificationPassManager =
      passBuilder.buildFunctionSimplificationPipeline(
          ::llvm::PassBuilder::O1, ::llvm::PassBuilder::ThinLTOPhase::PostLink,
          false);
  mThinLTOPreLinkPassManager =
      passBuilder.buildThinLTOPreLinkDefaultPipeline(::llvm::PassBuilder::O1);

  mModuleOptimizationPassManager.run(*mLLVMModule, moduleAnalysisManager);
  for (auto& func : *mLLVMModule) {
    if (!func.isDeclaration())
      mFunctionSimplificationPassManager.run(func, functionAnalysisManager);
  }
  // todo(abatashev) enable more aggressive optimizations once objects are
  // produced by chaos.
  // mThinLTOPreLinkPassManager.run(*mLLVMModule, moduleAnalysisManager);
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
