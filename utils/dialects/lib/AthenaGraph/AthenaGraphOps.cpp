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

#include "AthenaGraph/AthenaGraphOps.h"
#include "AthenaGraph/AthenaGraphDialect.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/OpImplementation.h"
#include "mlir/IR/StandardTypes.h"

namespace mlir::ath_graph {

void NodeOp::build(OpBuilder& builder, OperationState& result, StringRef name,
                   FunctionType type, size_t nodeId, size_t clusterId,
                   ArrayRef<NamedAttribute> attrs) {

  SmallVector<Type, 7> realArgTypes;
  std::copy(type.getInputs().begin(), type.getInputs().end(),
            std::back_inserter(realArgTypes));
  // Context pointer
  realArgTypes.push_back(builder.getIndexType());
  // Batch index
  realArgTypes.push_back(builder.getIndexType());

  auto realFuncType = builder.getFunctionType(realArgTypes, type.getResults());

  result.addAttribute(SymbolTable::getSymbolAttrName(),
                      builder.getStringAttr(name));
  result.addAttribute(getTypeAttrName(), TypeAttr::get(realFuncType));
  result.addAttributes(attrs);
  result.addAttribute(getNodeIdAttrName(),
                      IntegerAttr::get(builder.getIndexType(), nodeId));
  result.addAttribute(getClusterIdAttrName(),
                      IntegerAttr::get(builder.getIndexType(), clusterId));
  Region* body = result.addRegion();
  auto* entryBlock = new Block;
  entryBlock->addArguments(realFuncType.getInputs());

  body->getBlocks().push_back(entryBlock);
}

void GraphOp::build(OpBuilder& builder, OperationState& result, StringRef name,
                    ArrayRef<NamedAttribute> attrs) {
  llvm::SmallVector<mlir::Type, 2> graphArgs;
  // Context pointer
  graphArgs.push_back(builder.getIndexType());
  // Batch size
  graphArgs.push_back(builder.getIndexType());

  auto funcType = builder.getFunctionType(graphArgs, {});

  result.addAttribute(SymbolTable::getSymbolAttrName(),
                      builder.getStringAttr(name));
  result.addAttribute(getTypeAttrName(), TypeAttr::get(funcType));
  result.addAttributes(attrs);
  Region* body = result.addRegion();
  auto* entryBlock = new Block;
  entryBlock->addArguments(funcType.getInputs());

  body->getBlocks().push_back(entryBlock);
  ensureTerminator(*body, builder, result.location);
}

void SliceOp::build(OpBuilder& builder, OperationState& result, Value slice,
                    Value tensor) {
  result.addOperands(slice);
  result.addOperands(tensor);

  SmallVector<int64_t, 4> dims;
  auto tensorType = tensor.getType().cast<TensorType>();

  for (size_t i = 1; i < tensorType.getRank(); i++) {
    dims.push_back(tensorType.getDimSize(i));
  }

  result.addTypes(RankedTensorType::get(dims, tensorType.getElementType()));
}

void GetTensor::build(OpBuilder& builder, OperationState& result, Value context,
                      size_t virtualAddress, RankedTensorType type) {
  result.addOperands(context);
  // todo move attribute name.
  result.addAttribute("virtual_address", builder.getIndexAttr(virtualAddress));
  result.addTypes(type);
}

#define GET_OP_CLASSES
#include "AthenaGraph/AthenaGraphOps.cpp.inc"
} // namespace mlir::ath_graph
