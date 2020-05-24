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

#include "AthenaRuntime/AthenaRuntimeDialect.h"
#include "AthenaRuntime/AthenaRuntimeOps.h"
#include "Passes/Passes.h"

#include "mlir/Dialect/StandardOps/IR/Ops.h"
#include "mlir/IR/Module.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/DialectConversion.h"

using namespace mlir;

namespace {
class BarrierLegalizerPass
    : public PassWrapper<BarrierLegalizerPass, OperationPass<FuncOp>> {
protected:
  void runOnOperation() override {
    auto func = getOperation();
    auto module = func.getParentOfType<ModuleOp>();

    func.walk([&](ath_rt::BarrierOp barrier) {
      // todo replace with constant or function call
      auto clusterId = barrier.getAttrOfType<IntegerAttr>("cluster_id");

      SmallVector<Value, 8> dependants;
      func.walk([&](CallOp call) {
        auto calle = module.lookupSymbol<FuncOp>(call.getCallee());

        auto id = calle.getAttrOfType<IntegerAttr>("cluster_id");
        if (calle.getNumResults() == 1 && id && id == clusterId) {
          dependants.push_back(call.getResult(0));
        }
      });

      OpBuilder builder(module);
      builder.setInsertionPointAfter(barrier);
      builder.create<ath_rt::BarrierOp>(barrier.getLoc(), dependants);
      barrier.erase();
    });
  }
};
} // namespace

namespace mlir {

std::unique_ptr<OperationPass<FuncOp>> createBarrierLegalizerPass() {
  return std::make_unique<BarrierLegalizerPass>();
}
}
