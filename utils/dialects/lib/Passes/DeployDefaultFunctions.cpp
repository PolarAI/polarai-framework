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

#include "../utils/LaunchCommand.h"
#include "../utils/TensorInfo.h"
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
    // fixme byte is not always 8 bits
    auto voidPtrTy = LLVM::LLVMType::getInt8Ty(llvmDialect).getPointerTo();

    OpBuilder builder(module);
    builder.setInsertionPointToStart(module.getBody());
    {
      SmallVector<LLVM::LLVMType, 3> args;
      args.push_back(voidPtrTy); // GraphHandle
      args.push_back(voidPtrTy); // Device
      args.push_back(getTensorInfoType(llvmDialect).getPointerTo()); // Tensor

      auto funcTy = LLVM::LLVMType::getFunctionTy(
          LLVM::LLVMType::getVoidTy(llvmDialect), args, false);
      builder.create<mlir::LLVM::LLVMFuncOp>(builder.getUnknownLoc(),
                                             "ath_allocate", funcTy);
      builder.create<mlir::LLVM::LLVMFuncOp>(builder.getUnknownLoc(),
                                             "ath_release", funcTy);
    }
    {
      SmallVector<LLVM::LLVMType, 4> args;
      args.push_back(voidPtrTy); // GraphHandle
      args.push_back(voidPtrTy); // Device
      args.push_back(getTensorInfoType(llvmDialect).getPointerTo()); // Tensor
      args.push_back(LLVM::LLVMType::getInt32Ty(llvmDialect)); // Lock type

      auto funcTy = LLVM::LLVMType::getFunctionTy(
          LLVM::LLVMType::getVoidTy(llvmDialect), args, false);
      builder.create<mlir::LLVM::LLVMFuncOp>(builder.getUnknownLoc(),
                                             "ath_lock", funcTy);
    }
    {
      SmallVector<LLVM::LLVMType, 2> args;
      args.push_back(voidPtrTy);                               // GraphHandle
      args.push_back(LLVM::LLVMType::getInt64Ty(llvmDialect)); // NodeId

      auto funcTy = LLVM::LLVMType::getFunctionTy(voidPtrTy, args, false);
      builder.create<mlir::LLVM::LLVMFuncOp>(builder.getUnknownLoc(),
                                             "ath_device_select", funcTy);
    }

    {
      SmallVector<LLVM::LLVMType, 2> args;
      args.push_back(voidPtrTy);                               // GraphHandle
      args.push_back(LLVM::LLVMType::getInt64Ty(llvmDialect)); // NodeId
      args.push_back(getTensorInfoType(llvmDialect).getPointerTo()); // Tensor

      auto funcTy = LLVM::LLVMType::getFunctionTy(
          LLVM::LLVMType::getVoidTy(llvmDialect), args, false);
      builder.create<mlir::LLVM::LLVMFuncOp>(builder.getUnknownLoc(),
                                             "ath_load", funcTy);
    }
    {
      SmallVector<LLVM::LLVMType, 2> args;
      args.push_back(LLVM::LLVMType::getInt64Ty(llvmDialect)); // Count
      args.push_back(voidPtrTy.getPointerTo());                // Events

      auto funcTy = LLVM::LLVMType::getFunctionTy(
          LLVM::LLVMType::getVoidTy(llvmDialect), args, false);
      builder.create<mlir::LLVM::LLVMFuncOp>(builder.getUnknownLoc(),
                                             "ath_barrier", funcTy);
    }
    // fixme this piece of code must be autogenerated
    // LaunchCommand structure
    {
      auto launchCommandTy = getLaunchCommandType(llvmDialect);
      auto funcTy = LLVM::LLVMType::getFunctionTy(
          voidPtrTy,
          {voidPtrTy, voidPtrTy, voidPtrTy, launchCommandTy.getPointerTo()},
          false);

      builder.create<mlir::LLVM::LLVMFuncOp>(builder.getUnknownLoc(),
                                             "ath_launch", funcTy);
    }
  }
};
} // namespace

namespace mlir {
std::unique_ptr<OperationPass<ModuleOp>> createDeployDefaultFunctionsPass() {
  return std::make_unique<DeployDefaultFunctionsPass>();
}
} // namespace mlir
