//===----------------------------------------------------------------------===//
// Copyright (c) 2020 Polar. All rights reserved.
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

#include "Passes/Passes.h"
#include "PolarRuntime/PolarRuntimeDialect.h"
#include "PolarRuntime/PolarRuntimeOps.h"

#include "mlir/Conversion/AffineToStandard/AffineToStandard.h"
#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/GPU/GPUDialect.h"
#include "mlir/Dialect/SCF/SCF.h"
#include "mlir/Dialect/SPIRV/TargetAndABI.h"
#include "mlir/Dialect/StandardOps/IR/Ops.h"
#include "mlir/IR/BlockAndValueMapping.h"
#include "mlir/IR/Dominance.h"
#include "mlir/IR/Module.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/DialectConversion.h"
#include "mlir/Transforms/LoopUtils.h"

using namespace mlir;

namespace {
class KernelOutliningPass
    : public PassWrapper<KernelOutliningPass, OperationPass<ModuleOp>> {
protected:
  void runOnOperation() override {
    auto module = getOperation();

    module.walk([&module](polar_rt::LaunchOp launchOp) {
      DominanceInfo dominance(launchOp.getParentOfType<FuncOp>());
      OpBuilder builder(launchOp.getContext());

      auto parentNode = launchOp.getParentOfType<FuncOp>();
      auto kernelsModule = module.lookupSymbol<gpu::GPUModuleOp>("kernels");

      builder.setInsertionPointToStart(&kernelsModule.body().front());

      auto kernelName = parentNode.getName() + "_" + launchOp.kernel_name();

      llvm::SmallVector<mlir::Type, 5> argTypes;
      for (auto type : launchOp.body().front().getArgumentTypes()) {
        argTypes.push_back(type);
      }
      FunctionType funcType =
          FunctionType::get(argTypes, {}, launchOp.getContext());
      auto kernel = builder.create<gpu::GPUFuncOp>(launchOp.getLoc(),
                                                   kernelName.str(), funcType);
      kernel.setAttr(gpu::GPUDialect::getKernelFuncAttrName(),
                     builder.getUnitAttr());
      // todo should this be real local size?
      SmallVector<int, 3> localSize(3, 1);
      // todo replace these literals with constants
      kernel.setAttr("global_size", launchOp.getAttr("global_size"));
      kernel.setAttr("local_size", launchOp.getAttr("local_size"));
      kernel.setAttr(spirv::getEntryPointABIAttrName(),
                     spirv::getEntryPointABIAttr(localSize,
                                                 launchOp.getContext()));
      builder.setInsertionPointToStart(&kernel.body().front());

      BlockAndValueMapping mapping;

      for (int i = 0; i < launchOp.body().front().getNumArguments(); i++) {
        mapping.map(launchOp.body().front().getArgument(i),
                    kernel.body().front().getArgument(i));
      }

      for (auto& op : launchOp.body().front().without_terminator()) {
        for (auto operand : op.getOperands()) {
          if (dominance.dominates(operand, launchOp)) {
            auto clone = builder.clone(*operand.getDefiningOp(), mapping);
            mapping.map(operand, clone->getResult(0));
          }
        }
        auto clone = builder.clone(op, mapping);
        if (clone->getNumResults()) {
          for (auto res : llvm::enumerate(clone->getResults())) {
            mapping.map(op.getResult(res.index()), res.value());
          }
        }
      }

      builder.create<gpu::ReturnOp>(kernel.getLoc());

      builder.setInsertionPointAfter(launchOp);

      auto launchFuncOp = builder.create<polar_rt::LaunchFuncOp>(
          launchOp.getLoc(), builder.getSymbolRefAttr(kernel),
          launchOp.kernel_name(), launchOp.getOperands());

      launchOp.replaceAllUsesWith(launchFuncOp.getResult());
      launchOp.erase();
    });
  }
};
} // namespace

namespace mlir {
auto createKernelOutliningPass() -> std::unique_ptr<OperationPass<ModuleOp>> {
  return std::make_unique<KernelOutliningPass>();
}
} // namespace mlir
