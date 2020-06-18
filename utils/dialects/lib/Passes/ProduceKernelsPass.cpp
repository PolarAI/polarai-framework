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

#include "Passes/Passes.h"
#include "Compute/ComputeOps.h"
#include "AthenaGraph/ComputationalOpInterface.h"

#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/IR/Module.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/DialectConversion.h"

using namespace mlir;

namespace {
class ProduceKernelsPass
    : public PassWrapper<ProduceKernelsPass, OperationPass<FuncOp>> {

protected:
  void runOnOperation() override {
    auto func = getOperation();

    auto module = func.getParentOfType<ModuleOp>();
    auto kernelsModule = module.lookupSymbol<compute::ModuleOp>("kernels");

    func.walk([&](ath_graph::ComputationalOpInterface op) {
      if (!kernelsModule.lookupSymbol(op.getKernelName())) {
        op.produceKernel();
      }
    });
  }
};
} // namespace
namespace mlir {
auto createProduceKernelsPass() -> std::unique_ptr<OperationPass<FuncOp>> {
  return std::make_unique<ProduceKernelsPass>();
}
} // namespace mlir
