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

#include "AthenaGraph/AthenaGraphDialect.h"
#include "AthenaGraph/AthenaGraphOps.h"
#include "Passes/Passes.h"

#include "mlir/Dialect/StandardOps/IR/Ops.h"
#include "mlir/IR/Module.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/DialectConversion.h"

using namespace mlir;

namespace {
class LegalizeRTForLoweringPass
    : public PassWrapper<LegalizeRTForLoweringPass, OperationPass<FuncOp>> {

protected:
  void runOnOperation() override {
    auto func = getOperation();

    // fixme use more stable attr naming.
    func.walk([&](ath_graph::InvokeLoaderOp op) {
      op.setAttr("node_id", func.getAttr("node_id"));
    });
  }
};
} // namespace

namespace mlir {
auto createLegalizeRTForLoweringPass()
    -> std::unique_ptr<OperationPass<FuncOp>> {
  return std::make_unique<LegalizeRTForLoweringPass>();
}
} // namespace mlir
