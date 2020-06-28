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

#include <PolarRuntime/PolarRuntimeDialect.h>
#include <PolarRuntime/PolarRuntimeOps.h>

#include "mlir/IR/Builders.h"
#include "mlir/IR/OpImplementation.h"
#include "mlir/IR/StandardTypes.h"

namespace mlir::polar_rt {

void ScopeOp::build(OpBuilder& builder, OperationState& result, Value size) {
  result.addOperands(size);

  Region* bodyRegion = result.addRegion();
  auto* body = new Block();
  body->addArgument(IndexType::get(builder.getContext()));
  bodyRegion->push_back(body);
  ensureTerminator(*bodyRegion, builder, result.location);
}

#define GET_OP_CLASSES
#include "PolarRuntime/PolarRuntimeOps.cpp.inc"
} // namespace mlir::polar_rt
