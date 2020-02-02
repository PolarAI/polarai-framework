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

#ifndef ATHENA_STMT_H
#define ATHENA_STMT_H

#include "Decl.h"
#include "util.h"

#include <llvm/Support/raw_ostream.h>

#include <variant>

namespace chaos {
class DeclStmt {
private:
  AnyDecl mDecl;

public:
  DeclStmt() = delete;
  template <typename Decl>
  explicit DeclStmt(Decl decl) : mDecl(std::move(decl)){};
  DeclStmt(const DeclStmt&) = delete;
  DeclStmt(DeclStmt&&) = default;
  DeclStmt& operator=(const DeclStmt&) = delete;
  DeclStmt& operator=(DeclStmt&& stmt) = default;

  void dump(std::size_t tab = 0, bool isFirst = false) const {
    llvm::outs() << std::string(tab - isFirst, ' ') << (isFirst ? "`" : "")
                 << "|DeclStmt\n";
    impl::dump(mDecl, tab + 1, isFirst);
  };
};

class PassStmt {
public:
  PassStmt() = default;
  PassStmt(const PassStmt&) = delete;
  PassStmt(PassStmt&&) = default;
  PassStmt& operator=(const PassStmt&) = delete;
  PassStmt& operator=(PassStmt&&) noexcept = default;
  void dump(std::size_t tab = 0, bool isFirst = false) const {
    llvm::outs() << std::string(tab - isFirst, ' ') << (isFirst ? "`" : "")
                 << "|PassStmt\n";
  }
};
} // namespace chaos

#endif // ATHENA_STMT_H
