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

#pragma once

#include "Pattern.hpp"
#include "TypeConverter.hpp"

#include "clang/AST/ASTContext.h"
#include "llvm/ADT/ScopedHashTable.h"

namespace polarai::script {

class GenerationContext {
public:
  GenerationContext(clang::ASTContext* ctx,
                    std::unique_ptr<clang::MangleContext>& mangleCtx)
      : mContext(ctx), mMangleContext(mangleCtx){};

  auto getPatterns() -> PatternList& { return mPatterns; }
  auto getTypeConverter() -> TypeConverter& { return mTypeConverter; }
  auto getSymbolTable()
      -> llvm::ScopedHashTable<llvm::StringRef, mlir::Value>& {
    return mSymbolTable;
  }

private:
  clang::ASTContext* mContext;
  std::unique_ptr<clang::MangleContext>& mMangleContext;
  PatternList mPatterns{};
  TypeConverter mTypeConverter;
  llvm::ScopedHashTable<llvm::StringRef, mlir::Value> mSymbolTable;
};
} // namespace polarai::script
