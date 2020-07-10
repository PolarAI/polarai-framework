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

#include "mlir/Analysis/SliceAnalysis.h"
#include "mlir/Dialect/StandardOps/IR/Ops.h"
#include "mlir/IR/Module.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/DialectConversion.h"
#include "llvm/ADT/SetVector.h"

using namespace mlir;

namespace {
class RuntimeShapeInferencePass
    : public PassWrapper<RuntimeShapeInferencePass, OperationPass<FuncOp>> {
protected:
  void runOnOperation() override {
    // todo right now memviews are not supported
    auto func = getOperation();

    func.walk([](polar_rt::ApplyOp applyOp) {
      auto allArgs = applyOp.getOperands();
      auto args =
          polar_rt::ApplyOp::operand_range(allArgs.begin() + 2, allArgs.end());

      for (auto arg : llvm::enumerate(args)) {
        if (arg.value().getType().isa<RankedTensorType>()) {
          auto tensorTy = arg.value().getType().cast<RankedTensorType>();
          auto memrefArg = applyOp.body().front().getArgument(arg.index());
          memrefArg.setType(
              MemRefType::get(tensorTy.getShape(), tensorTy.getElementType()));
        }
      }
    });
  }
};
} // namespace

namespace mlir {
auto createRuntimeShapeInferencePass()
    -> std::unique_ptr<OperationPass<FuncOp>> {
  return std::make_unique<RuntimeShapeInferencePass>();
}
} // namespace mlir
