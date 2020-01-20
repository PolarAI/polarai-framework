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
#ifndef ATHENA_LLVMTOSTDPASS_H
#define ATHENA_LLVMTOSTDPASS_H

#include "LLVMToStandardTypeConverter.h"
#include <mlir/Pass/Pass.h>

namespace chaos {
class LLVMToStdPass : public mlir::ModulePass<LLVMToStdPass> {
private:
  LLVMToStandardTypeConverter mTypeConverter;

public:
  LLVMToStdPass(mlir::MLIRContext* ctx)
      : mTypeConverter(new mlir::StandardOpsDialect(ctx)){};

  void runOnModule() override;
};

std::unique_ptr<LLVMToStdPass> createLLVMToStdPass(mlir::MLIRContext* ctx);

} // namespace chaos

#endif // ATHENA_LLVMTOSTDPASS_H
