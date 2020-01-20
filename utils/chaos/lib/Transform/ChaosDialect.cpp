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

#include "mlir/IR/Builders.h"
#include "mlir/IR/StandardTypes.h"

#include "ChaosDialect.h"
chaos::ChaosDialect::ChaosDialect(mlir::MLIRContext* ctx)
    : mlir::Dialect("chaos", ctx) {
  addOperations<
#define GET_OP_LIST
#include "ChaosDialect.cpp.inc"
      >();
}

void chaos::ReinterpretOp::build(mlir::Builder* builder,
                                 mlir::OperationState& state,
                                 mlir::Value source, mlir::Type destType) {
  state.addTypes(destType);
  state.addOperands(source);
}

namespace chaos {
using namespace llvm;
#define GET_OP_CLASSES
#include "ChaosDialect.cpp.inc"
} // namespace chaos