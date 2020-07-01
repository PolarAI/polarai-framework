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

#include "Compute/ComputeDialect.h"
#include "Compute/ComputeOps.h"
#include "Passes/Passes.h"
#include "PolarRuntime/PolarRuntimeDialect.h"
#include "PolarRuntime/PolarRuntimeOps.h"

#include "mlir/Conversion/AffineToStandard/AffineToStandard.h"
#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/SCF/SCF.h"
#include "mlir/Dialect/StandardOps/IR/Ops.h"
#include "mlir/IR/BlockAndValueMapping.h"
#include "mlir/IR/Module.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/DialectConversion.h"
#include "mlir/Transforms/LoopUtils.h"

using namespace mlir;

namespace {
class KernelMaterializerPass
    : public PassWrapper<KernelMaterializerPass, OperationPass<FuncOp>> {
protected:
  void runOnOperation() override {
    auto func = getOperation();

    func.walk([](polar_rt::ApplyOp applyOp) {
      OpBuilder builder(applyOp.getContext());
      builder.setInsertionPointAfter(applyOp);

      auto& body = applyOp.body().front();
      auto& cand = body.getOperations().front();

      if (!llvm::isa<scf::ForOp>(cand)) {
        return;
      }

      auto affineFor = llvm::cast<scf::ForOp>(cand);

      SmallVector<scf::ForOp, 3> loopNest;
      getPerfectlyNestedLoops(loopNest, affineFor);

      if (loopNest.size() > 3) {
        return;
      }

      SmallVector<int64_t, 3> globalOffset;
      SmallVector<int64_t, 3> globalSize;
      SmallVector<int64_t, 3> localSize(3, 0);

      for (auto& loop : llvm::enumerate(loopNest)) {
        auto lbOp = loop.value().lowerBound().getDefiningOp();
        auto ubOp = loop.value().upperBound().getDefiningOp();

        if (!llvm::isa<ConstantIndexOp>(lbOp) &&
            !llvm::isa<ConstantIndexOp>(ubOp)) {
          llvm::errs() << "Crap\n";
          return;
        }

        int64_t lb = llvm::cast<ConstantIndexOp>(lbOp).getValue();
        int64_t ub = llvm::cast<ConstantIndexOp>(ubOp).getValue();

        globalOffset.push_back(lb);
        globalSize.push_back(ub - lb);
      }

      SmallVector<Type, 5> blockArgTypes;
      for (auto type : applyOp.body().front().getArgumentTypes()) {
        blockArgTypes.push_back(type);
      }
      auto launchOp = builder.create<polar_rt::LaunchOp>(
          applyOp.getLoc(), applyOp.kernel_name(), applyOp.getOperands(),
          blockArgTypes, globalOffset, globalSize, localSize);

      OpBuilder bodyBuilder(applyOp.getContext());
      bodyBuilder.setInsertionPointToStart(&launchOp.body().front());
      BlockAndValueMapping mapping;
      for (auto& loop : llvm::enumerate(loopNest)) {
        auto dim = bodyBuilder.create<compute::GlobalIdOp>(
            loop.value().getLoc(), bodyBuilder.getIndexType(),
            bodyBuilder.getIndexAttr(loop.index()));
        mapping.map(loop.value().getInductionVar(), dim);
      }

      for (int i = 0; i < launchOp.body().front().getNumArguments(); i++) {
        mapping.map(applyOp.body().front().getArgument(i),
                    launchOp.body().front().getArgument(i));
      }
      for (auto& op : loopNest.back().getBody()->without_terminator()) {
        auto clone = bodyBuilder.clone(op, mapping);
        if (clone->getNumResults()) {
          for (auto res : llvm::enumerate(clone->getResults())) {
            mapping.map(op.getResult(res.index()), res.value());
          }
        }
      }

      applyOp.replaceAllUsesWith(launchOp.getResult());
      applyOp.erase();
    });
  }
};
} // namespace

namespace mlir {
auto createKernelMaterializerPass() -> std::unique_ptr<OperationPass<FuncOp>> {
  return std::make_unique<KernelMaterializerPass>();
}
} // namespace mlir
