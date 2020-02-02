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

#ifndef ATHENA_DECL_H
#define ATHENA_DECL_H

#include "../Lexer.h"
#include "ForwardDeclarations.h"
#include "util.h"

#include <llvm/Support/raw_ostream.h>

#include <utility>
#include <variant>
#include <vector>

namespace chaos {
class TranslationUnitDecl {
protected:
  std::vector<AnyStatement> mStatements;

  template <typename Stmt> void addStatement(Stmt stmt) {
    mStatements.push_back(std::move(stmt));
  }

  friend class Parser;

public:
  TranslationUnitDecl() = default;
  TranslationUnitDecl(const TranslationUnitDecl&) = delete;
  TranslationUnitDecl(TranslationUnitDecl&&) = default;
  TranslationUnitDecl& operator=(const TranslationUnitDecl&) = delete;
  TranslationUnitDecl& operator=(TranslationUnitDecl&& rhs) = default;

  void dump(std::size_t tab = 0, bool isFirst = false) const;
};

class BlockDecl {
protected:
  std::vector<AnyStatement> mStatements;

  void addStatement(AnyStatement stmt);

  friend class Parser;

public:
  BlockDecl() = default;
  BlockDecl(const BlockDecl&) = delete;
  BlockDecl(BlockDecl&&) = default;
  BlockDecl& operator=(const BlockDecl&) = delete;
  BlockDecl& operator=(BlockDecl&& rhs) = default;
  void dump(std::size_t tab = 0, bool isFirst = false) const;
};

class FunctionDecl {
protected:
  std::string mFunctionName;
  SourceLocation mLocation;
  std::vector<ParamDecl> mParams;
  BlockDecl mBody;

  void addParam(ParamDecl paramDecl);
  void setBody(BlockDecl blockDecl);

  friend class Parser;

public:
  FunctionDecl() = delete;
  FunctionDecl(std::string name, SourceLocation loc)
      : mFunctionName(std::move(name)), mLocation(std::move(loc)) {}
  FunctionDecl(const FunctionDecl&) = delete;
  FunctionDecl(FunctionDecl&&) = default;
  FunctionDecl& operator=(const FunctionDecl&) = delete;
  FunctionDecl& operator=(FunctionDecl&& rhs) = default;

  void dump(std::size_t tab = 0, bool isFirst = false) const;
};

class ParamDecl {
protected:
  std::string mParamName;
  std::string mType;
  SourceLocation mLocation;

public:
  ParamDecl() = delete;
  ParamDecl(std::string name, std::string type, SourceLocation loc)
      : mParamName(std::move(name)), mType(std::move(type)),
        mLocation(std::move(loc)) {}
  ParamDecl(const ParamDecl&) = delete;
  ParamDecl(ParamDecl&&) = default;
  ParamDecl& operator=(const ParamDecl&) = delete;
  ParamDecl& operator=(ParamDecl&& rhs) = default;

  void dump(std::size_t tab = 0, bool isFirst = false) const;
};
} // namespace chaos

#endif // ATHENA_DECL_H
