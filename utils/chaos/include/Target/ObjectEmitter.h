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

#ifndef ATHENA_OBJECTEMITTER_H
#define ATHENA_OBJECTEMITTER_H

#include <Target/export.h>
#include <llvm/IR/Module.h>
#include <string>

class CHAOS_TARGET_EXPORT ObjectEmitter {
private:
  llvm::LLVMContext mContext;
  std::vector<std::unique_ptr<llvm::MemoryBuffer>> mLinkableModules;

public:
  ObjectEmitter();
  void addModule(const std::string& filename);
  void emitObject(const std::string& filename);
};
#endif // ATHENA_OBJECTEMITTER_H
