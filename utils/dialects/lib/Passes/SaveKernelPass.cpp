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
#include <athena/backend/llvm/runtime/ProgramDesc.h>

#include "mlir/Conversion/AffineToStandard/AffineToStandard.h"
#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/GPU/GPUDialect.h"
#include "mlir/Dialect/SCF/SCF.h"
#include "mlir/Dialect/StandardOps/IR/Ops.h"
#include "mlir/IR/BlockAndValueMapping.h"
#include "mlir/IR/Dominance.h"
#include "mlir/IR/Module.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/DialectConversion.h"
#include "mlir/Transforms/LoopUtils.h"

#include <fstream>

using namespace mlir;

namespace {
class SaveKernelPass
    : public PassWrapper<SaveKernelPass, OperationPass<gpu::GPUModuleOp>> {
public:
  SaveKernelPass(SaveKernelCallback callback)
      : mCallback(std::move(callback)) {}

protected:
  void runOnOperation() override {
    auto module = getOperation();

    auto ptx = module.getAttrOfType<StringAttr>("nvvm.ptx");

    athena::backend::llvm::ProgramDesc desc;
    desc.type = athena::backend::llvm::ProgramDesc::ProgramType::PTX;
    desc.data = std::vector(ptx.getValue().begin(), ptx.getValue().end());

    mCallback(desc);
  }

private:
  SaveKernelCallback mCallback;
};
} // namespace

namespace mlir {
auto createSaveKernelPass(SaveKernelCallback callback)
    -> std::unique_ptr<OperationPass<gpu::GPUModuleOp>> {
  return std::make_unique<SaveKernelPass>(std::move(callback));
}
} // namespace mlir
