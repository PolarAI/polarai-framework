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

void ApplyOp::build(OpBuilder& builder, OperationState& result,
                    Value device, Value event,
                    StringRef kernelName, ValueRange operands) {
  result.addOperands(device);
  result.addOperands(event);
  result.addOperands(operands);
  result.types.push_back(EventType::get(builder.getContext()));
  result.addAttribute("kernel_name", builder.getStringAttr(kernelName));

  SmallVector<Type, 5> blockArgTypes;
  for (auto op : operands) {
    auto type = op.getType();
    if (type.isa<RankedTensorType>()) {
      auto tensorTy = type.cast<RankedTensorType>();
      SmallVector<int64_t, 3> dims(tensorTy.getRank(), -1);
      blockArgTypes.push_back(MemRefType::get(dims, tensorTy.getElementType()));
    } else {
      blockArgTypes.push_back(type);
    }
  }

  auto* body = new Block;
  body->addArguments(blockArgTypes);

  Region* kernelRegion = result.addRegion();
  kernelRegion->push_back(body);

  OpBuilder::InsertionGuard guard{builder};
  builder.setInsertionPointToStart(body);
  builder.create<TerminatorOp>(builder.getUnknownLoc());
}

void LaunchOp::build(OpBuilder& builder, OperationState& result,
                     StringRef kernelName, ValueRange operands,
                     ArrayRef<Type> blockArgTypes,
                     ArrayRef<int64_t> globalOffset,
                     ArrayRef<int64_t> globalSize,
                     ArrayRef<int64_t> localSize) {
  result.addOperands(operands);
  result.addAttribute("kernel_name", builder.getStringAttr(kernelName));
  result.addAttribute("global_offset", builder.getI64ArrayAttr(globalOffset));
  result.addAttribute("global_size", builder.getI64ArrayAttr(globalSize));
  result.addAttribute("local_size", builder.getI64ArrayAttr(localSize));
  result.types.push_back(EventType::get(builder.getContext()));

  auto* body = new Block;
  body->addArguments(blockArgTypes);

  Region* kernelRegion = result.addRegion();

  OpBuilder::InsertionGuard guard{builder};
  builder.setInsertionPointToStart(body);
  builder.create<TerminatorOp>(builder.getUnknownLoc());
  kernelRegion->push_back(body);
}

void LaunchFuncOp::build(OpBuilder& builder, OperationState& result,
                     SymbolRefAttr kernel,
                     StringRef nativeKernel, ValueRange operands) {
  result.addOperands(operands);
  result.addAttribute("kernel", kernel);
  result.addAttribute("native_kernel", builder.getStringAttr(nativeKernel));
  result.types.push_back(EventType::get(builder.getContext()));
}

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
