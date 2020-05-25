/*
 * Copyright (c) 2019 Athena. All rights reserved.
 * https://getathena.ml
 *
 * Licensed under MIT license.
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS, WITHOUT
 * WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the
 * License for the specific language governing permissions and limitations under
 * the License.
 */

#ifndef ATHENA_GRAPHDIALECT_H
#define ATHENA_GRAPHDIALECT_H

#include "mlir/IR/Dialect.h"
#include "mlir/IR/Function.h"

#include <athena/core/tensor/impl/TensorImpl.h>

namespace athena::backend::llvm {
class GraphDialect : public mlir::Dialect {
public:
  explicit GraphDialect(mlir::MLIRContext* ctx);

  /// Provide a utility accessor to the dialect namespace. This is used by
  /// several utilities for casting between dialects.
  static ::llvm::StringRef getDialectNamespace() { return "graph"; }
};

using namespace mlir; // WTF, Google?
#define GET_OP_CLASSES
#include "GraphDialect.h.inc"

} // namespace athena::backend::llvm

#endif // ATHENA_GRAPHDIALECT_H
