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

static void getChildren(Operation* op, llvm::SetVector<Operation*>& children) {
  if (op->getNumResults() == 0) {
    return;
  }
  Value res = op->getResult(0);
  for (auto u : res.getUsers()) {
    if (llvm::isa<polar_rt::LaunchFuncOp>(u)) {
      children.insert(u);
      getChildren(u, children);
    }
  }
}

namespace {
class ReleaseDependencyPass
    : public PassWrapper<ReleaseDependencyPass, OperationPass<FuncOp>> {
protected:
  void runOnOperation() override {
    auto func = getOperation();

    func.walk([&](polar_rt::ReleaseOp op) {
      auto* tensor = op.tensor().getDefiningOp();
      llvm::SetVector<Operation*> children;
      getChildren(tensor, children);

      if (children.empty()) {
        return;
      }

      auto sortedTensors = mlir::topologicalSort(children);

      auto lastLaunch =
          llvm::cast<polar_rt::LaunchFuncOp>(sortedTensors.back());
      op.eventMutable().append(lastLaunch.out_event());
    });
  }
};
} // namespace

namespace mlir {
auto createReleaseDependencyPass() -> std::unique_ptr<OperationPass<FuncOp>> {
  return std::make_unique<ReleaseDependencyPass>();
}
} // namespace mlir
