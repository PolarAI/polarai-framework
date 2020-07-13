//===----------------------------------------------------------------------===//
// Copyright (c) 2020 PolarAI. All rights reserved.
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
#include <polarai/backend/generic/runtime/ProgramDesc.hpp>

#include "mlir/Conversion/AffineToStandard/AffineToStandard.h"
#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/GPU/GPUDialect.h"
#include "mlir/Dialect/SCF/SCF.h"
#include "mlir/Dialect/SPIRV/SPIRVOps.h"
#include "mlir/Dialect/SPIRV/Serialization.h"
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
    : public PassWrapper<SaveKernelPass, OperationPass<ModuleOp>> {
public:
  SaveKernelPass(SaveKernelCallback callback)
      : mCallback(std::move(callback)) {}

protected:
  void runOnOperation() override {
    auto module = getOperation();

    auto fillKernels = [](auto func, polarai::backend::generic::ProgramDesc& desc) {
      polarai::backend::generic::KernelDesc kernel;
      // todo use constant
      auto gs = func.template getAttrOfType<ArrayAttr>("global_size");
      auto gsArr = gs.getValue();
      kernel.globalX = gsArr[0].template cast<IntegerAttr>().getInt();
      kernel.globalY = gsArr[1].template cast<IntegerAttr>().getInt();
      kernel.globalZ = gsArr[2].template cast<IntegerAttr>().getInt();

      auto ls = func.template getAttrOfType<ArrayAttr>("local_size");
      auto lsArr = ls.getValue();
      kernel.localX = lsArr[0].template cast<IntegerAttr>().getInt();
      kernel.localY = lsArr[1].template cast<IntegerAttr>().getInt();
      kernel.localZ = lsArr[2].template cast<IntegerAttr>().getInt();

      desc.kernels[func.getName().str()] = kernel;
    };

    module.walk([this, fillKernels](gpu::GPUModuleOp module) {
      auto ptx = module.getAttrOfType<StringAttr>("nvvm.ptx");

      polarai::backend::generic::ProgramDesc desc;
      desc.type = polarai::backend::generic::ProgramDesc::Type::PTX;
      desc.data = std::vector(ptx.getValue().begin(), ptx.getValue().end());

      module.walk([&](gpu::GPUFuncOp func) { fillKernels(func, desc); });

      mCallback(desc);
    });

    module.walk([this, fillKernels](spirv::ModuleOp module) {
      SmallVector<uint32_t, 4096> binary;
      spirv::serialize(module, binary, false);

      auto begin = reinterpret_cast<char*>(binary.data());
      std::vector<char> data(begin, begin + binary.size() * sizeof(uint32_t));

      polarai::backend::generic::ProgramDesc desc;
      desc.type = polarai::backend::generic::ProgramDesc::Type::SPIRV_SHADER;
      desc.data = std::move(data);

      module.walk([&](spirv::FuncOp func) {
        fillKernels(func, desc);
      });

      mCallback(desc);
    });
  }

private:
  SaveKernelCallback mCallback;
};
} // namespace

namespace mlir {
auto createSaveKernelPass(SaveKernelCallback callback)
    -> std::unique_ptr<OperationPass<ModuleOp>> {
  return std::make_unique<SaveKernelPass>(std::move(callback));
}
} // namespace mlir
