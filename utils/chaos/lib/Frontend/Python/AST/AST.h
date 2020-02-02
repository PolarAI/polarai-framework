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

#ifndef ATHENA_AST_H
#define ATHENA_AST_H

#include "../Lexer.h"
#include "Decl.h"
#include "Stmt.h"
#include <llvm/Support/Debug.h>
#include <llvm/Support/raw_ostream.h>
#include <memory>
#include <string>
#include <utility>
#include <variant>
#include <vector>

namespace chaos {
class AST {
private:
  std::unique_ptr<TranslationUnitDecl> mTranslationUnit{};

public:
  explicit AST(std::unique_ptr<TranslationUnitDecl> translationUnit)
      : mTranslationUnit(std::move(translationUnit)) {}
  TranslationUnitDecl& getTranslationUnit() { return *mTranslationUnit; }

  void dump() const { mTranslationUnit->dump(); };
};

} // namespace chaos

#endif // ATHENA_AST_H
