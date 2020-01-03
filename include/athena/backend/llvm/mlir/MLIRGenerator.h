/*
 * Copyright (c) 2019 Athena. All rights reserved.
 * https://getathena.ml
 *
 * Licensed under MIT license.
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS, WITHOUT
 * WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the
 * License for the specific language governing permissions and limitations under
 * the License.
 */

#ifndef ATHENA_MLIRGENERATOR_H
#define ATHENA_MLIRGENERATOR_H

#include "mlir/IR/Attributes.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/Function.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/Module.h"
#include "mlir/IR/StandardTypes.h"

#include <athena/backend/llvm/mlir/mlir_export.h>
#include <athena/core/AbstractGenerator.h>

#include <map>

namespace athena::backend::llvm {
/// The old Generator interface is not suitable for MLIR. This is for testing
/// only.
class ATH_MLIR_EXPORT MLIRGenerator : public core::AbstractGenerator {
private:
  mlir::MLIRContext mContext;
  mlir::ModuleOp mModule;
  mlir::OpBuilder mBuilder;
  std::map<size_t, mlir::Value> mTensorValueMap;

public:
  MLIRGenerator();
  void openNode(std::string_view name) override;
  void closeNode() override;
  void generateFunctionHeader(const std::string& name) override;
  void generateFunctionFooter() override;
  void generateLoad(const core::AbstractLoader& loader,
                    core::inner::Tensor& tensor) override;
  ~MLIRGenerator() = default;

  mlir::ModuleOp& getModule() { return mModule; }

protected:
  void generateImpl(std::string& string, core::inner::Tensor& a) override;
  void generateImpl(std::string& string, core::inner::Tensor& a,
                    void*& b) override;
  void generateImpl(std::string& string, core::inner::Tensor& a,
                    core::inner::Tensor& b) override;
  void generateImpl(std::string& string, core::inner::Tensor& a,
                    core::inner::Tensor& b, core::inner::Tensor& c) override;
  void generateImpl(std::string& string, core::inner::Tensor& a,
                    uint64_t scaleA, core::inner::Tensor& b, uint64_t scaleB,
                    core::inner::Tensor& c) override;
  void generateImpl(std::string& string, void* pVoid, core::inner::Tensor& a,
                    core::inner::Tensor& b, core::inner::Tensor& c) override;
};
} // namespace athena::backend::llvm

#endif // ATHENA_MLIRGENERATOR_H
