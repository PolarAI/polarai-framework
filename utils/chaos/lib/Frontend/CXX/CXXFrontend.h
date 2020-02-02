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
#ifndef ATHENA_CXXFRONTEND_H
#define ATHENA_CXXFRONTEND_H

#include "clang/Frontend/CompilerInstance.h"
#include <llvm/Option/Option.h>

namespace chaos {
class CXXFrontend {
private:
  std::unique_ptr<clang::CompilerInstance> mCompilerInstance;

public:
  CXXFrontend();

  void run(std::string_view filename, const std::vector<std::string>& args);
};
} // namespace chaos
#endif // ATHENA_CXXFRONTEND_H
