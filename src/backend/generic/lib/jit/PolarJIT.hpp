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

#include <llvm/ExecutionEngine/Orc/LLJIT.h>
#include <llvm/Support/FileSystem.h>
#include <mlir/IR/Module.h>
#include <mlir/Pass/PassManager.h>

namespace polarai::backend::generic {
class Device;
struct ProgramDesc;
class PolarJIT {
public:
  PolarJIT(std::unique_ptr<::llvm::orc::LLJIT> jit);

  static auto create() -> std::shared_ptr<PolarJIT>;
  static auto createWithDebugging() -> std::shared_ptr<PolarJIT>;

  void addModule(const mlir::OwningModuleRef& ref);
  auto lookupSymbol(::llvm::StringRef symbolName) -> ::llvm::JITTargetAddress;

  auto getContext() -> mlir::MLIRContext* { return &mContext; }

  void registerDevice(std::shared_ptr<Device>);
  void resetDevices();

private:
  void setupMlirPassManager();
  void compileModule();

  mlir::MLIRContext mContext;
  mlir::PassManager mMlirPassManager;
  mlir::OwningModuleRef mInternalModule;
  std::unique_ptr<::llvm::orc::LLJIT> mJITInstance;
#ifdef DEBUG
  ::llvm::SmallVector<char, 128> mTempFileGraph;
#endif
  std::vector<std::shared_ptr<Device>> mRegisteredDevices;
  std::vector<std::shared_ptr<ProgramDesc>> mCompiledPrograms;
};
} // namespace polarai::backend::llvm
