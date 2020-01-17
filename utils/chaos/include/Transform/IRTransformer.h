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

#ifndef ATHENA_IRTRANSFORMER_H
#define ATHENA_IRTRANSFORMER_H

#include <Transform/export.h>
#include <llvm/IR/Module.h>
#include <mlir/IR/Module.h>
#include <string>

namespace chaos {
class CHAOS_TRANSFORM_EXPORT IRTransformer {
private:
  std::unique_ptr<llvm::LLVMContext> mLLVMContext;
  mlir::MLIRContext mMLIRContext;
  std::unique_ptr<mlir::OwningModuleRef> mMLIRModule;
  std::unique_ptr<llvm::Module> mLLVMModule;
  llvm::DataLayout mDataLayout;

public:
  explicit IRTransformer(std::unique_ptr<llvm::Module> llvmModule,
                         std::unique_ptr<llvm::LLVMContext> ctx);
  static std::unique_ptr<IRTransformer>
  getFromIrFile(const std::string& filename);

  void run();

  void dumpMLIR(const std::string& filename);
  void dumpLLVMIR(const std::string& filename);
  void writeBitcode(const std::string& filename);
};
} // namespace chaos

#endif // ATHENA_IRTRANSFORMER_H
