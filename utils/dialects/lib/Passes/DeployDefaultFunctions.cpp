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

#include "LoaderFunctionAnalysis.h"
#include "Passes/Passes.h"

#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/IR/Module.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/DialectConversion.h"

using namespace mlir;

namespace {
class DeployDefaultFunctionsPass
    : public PassWrapper<DeployDefaultFunctionsPass, OperationPass<ModuleOp>> {

protected:
  void runOnOperation() override {
    auto module = getOperation();
    auto* llvmDialect = getContext().getRegisteredDialect<LLVM::LLVMDialect>();

    OpBuilder builder(module);
    builder.setInsertionPointToStart(module.getBody());
    {
      SmallVector<LLVM::LLVMType, 3> args;
      args.push_back(LLVM::LLVMType::getVoidTy(llvmDialect).getPointerTo());
      args.push_back(LLVM::LLVMType::getVoidTy(llvmDialect).getPointerTo());
      args.push_back(LLVM::LLVMType::getVoidTy(llvmDialect).getPointerTo());

      auto funcTy = LLVM::LLVMType::getFunctionTy(
          LLVM::LLVMType::getVoidTy(llvmDialect), args, false);
      builder.create<mlir::LLVM::LLVMFuncOp>(builder.getUnknownLoc(),
                                             "ath_allocate", funcTy);
      builder.create<mlir::LLVM::LLVMFuncOp>(builder.getUnknownLoc(),
                                             "ath_release_tensor", funcTy);
    }
    {
      SmallVector<LLVM::LLVMType, 3> args;
      args.push_back(LLVM::LLVMType::getVoidTy(llvmDialect).getPointerTo());
      args.push_back(LLVM::LLVMType::getVoidTy(llvmDialect).getPointerTo());
      args.push_back(
          LLVM::LLVMType::getInt64Ty(llvmDialect)); // fixme 32-bit systems?

      auto funcTy = LLVM::LLVMType::getFunctionTy(
          LLVM::LLVMType::getVoidTy(llvmDialect).getPointerTo(), args, false);
      builder.create<mlir::LLVM::LLVMFuncOp>(builder.getUnknownLoc(),
                                             "ath_get_tensor_ptr", funcTy);
    }
    {
      SmallVector<LLVM::LLVMType, 4> args;
      args.push_back(LLVM::LLVMType::getVoidTy(llvmDialect).getPointerTo());
      args.push_back(LLVM::LLVMType::getVoidTy(llvmDialect).getPointerTo());
      args.push_back(LLVM::LLVMType::getVoidTy(llvmDialect).getPointerTo());
      args.push_back(LLVM::LLVMType::getInt32Ty(llvmDialect));

      auto funcTy = LLVM::LLVMType::getFunctionTy(
          LLVM::LLVMType::getVoidTy(llvmDialect), args, false);
      builder.create<mlir::LLVM::LLVMFuncOp>(builder.getUnknownLoc(),
                                             "ath_lock_tensor", funcTy);
      {
        SmallVector<LLVM::LLVMType, 2> args;
        args.push_back(
            LLVM::LLVMType::getInt64Ty(llvmDialect)); // fixme sizeof size_t
        args.push_back(LLVM::LLVMType::getVoidTy(llvmDialect).getPointerTo());
        args.push_back(LLVM::LLVMType::getInt32Ty(llvmDialect));

        auto funcTy = LLVM::LLVMType::getFunctionTy(
            LLVM::LLVMType::getVoidTy(llvmDialect).getPointerTo(), args, false);
        builder.create<mlir::LLVM::LLVMFuncOp>(builder.getUnknownLoc(),
                                               "ath_get_sub_tensor", funcTy);
        builder.create<mlir::LLVM::LLVMFuncOp>(
            builder.getUnknownLoc(), "ath_get_device_for_node", funcTy);
      }
    }

    {
      SmallVector<LLVM::LLVMType, 3> args;
      args.push_back(LLVM::LLVMType::getVoidTy(llvmDialect).getPointerTo());
      args.push_back(LLVM::LLVMType::getVoidTy(llvmDialect).getPointerTo());
      args.push_back(LLVM::LLVMType::getVoidTy(llvmDialect).getPointerTo());

      auto funcTy = LLVM::LLVMType::getFunctionTy(
          LLVM::LLVMType::getVoidTy(llvmDialect), args, false);

      auto loaderFuncAnalysis = getAnalysis<LoaderFunctionAnalysis>();
      auto funcNames = loaderFuncAnalysis.getLoaderFunctionNames();

      for (const auto& funcName : funcNames) {
        builder.create<mlir::LLVM::LLVMFuncOp>(builder.getUnknownLoc(),
                                               funcName, funcTy);
      }
    }
  }
};
} // namespace

namespace mlir {
std::unique_ptr<OperationPass<ModuleOp>> createDeployDefaultFunctionsPass() {
  return std::make_unique<DeployDefaultFunctionsPass>();
}
} // namespace mlir