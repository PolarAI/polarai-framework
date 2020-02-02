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

#ifndef ATHENA_CLANGDIALECT_H
#define ATHENA_CLANGDIALECT_H

#include "mlir/IR/Dialect.h"
#include "mlir/IR/Function.h"

namespace clang {
class ClangDialect : public mlir::Dialect {
public:
  explicit ClangDialect(mlir::MLIRContext* ctx);

  /// Provide a utility accessor to the dialect namespace. This is used by
  /// several utilities for casting between dialects.
  static llvm::StringRef getDialectNamespace() { return "clang"; }
};

using namespace mlir;
#define GET_OP_CLASSES
#include <Dialects/ClangDialect.h.inc>
} // namespace clang

#endif // ATHENA_CLANGDIALECT_H
