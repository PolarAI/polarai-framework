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
#include "mlir/IR/FunctionImplementation.h"
#include "mlir/IR/OpImplementation.h"
#include "mlir/IR/StandardTypes.h"

namespace mlir::ath_graph {

void NodeOp::build(OpBuilder& builder, OperationState& result, StringRef name,
                   FunctionType type, size_t nodeId, size_t clusterId,
                   ArrayRef<NamedAttribute> attrs) {

  result.addAttribute(SymbolTable::getSymbolAttrName(),
                      builder.getStringAttr(name));
  result.addAttribute(getTypeAttrName(), TypeAttr::get(type));
  result.addAttributes(attrs);
  result.addAttribute(getNodeIdAttrName(),
                      IntegerAttr::get(builder.getIndexType(), nodeId));
  result.addAttribute(getClusterIdAttrName(),
                      IntegerAttr::get(builder.getIndexType(), clusterId));
  Region* body = result.addRegion();
  auto* entryBlock = new Block;
  entryBlock->addArguments(type.getInputs());

  body->getBlocks().push_back(entryBlock);
}

static ParseResult
parseAttributions(OpAsmParser& parser, StringRef keyword,
                  SmallVectorImpl<OpAsmParser::OperandType>& args,
                  SmallVectorImpl<Type>& argTypes) {
  if (failed(parser.parseOptionalKeyword(keyword)))
    return success();

  if (failed(parser.parseLParen()))
    return failure();

  // Early exit for an empty list.
  if (succeeded(parser.parseOptionalRParen()))
    return success();

  do {
    OpAsmParser::OperandType arg;
    Type type;

    if (parser.parseRegionArgument(arg) || parser.parseColonType(type))
      return failure();

    args.push_back(arg);
    argTypes.push_back(type);
  } while (succeeded(parser.parseOptionalComma()));

  return parser.parseRParen();
}

static void printAttributions(OpAsmPrinter& p, StringRef keyword,
                              ArrayRef<BlockArgument> values) {
  if (values.empty())
    return;

  p << ' ' << keyword << '(';
  llvm::interleaveComma(
      values, p, [&p](BlockArgument v) { p << v << " : " << v.getType(); });
  p << ')';
}

static ParseResult parseNodeOp(OpAsmParser& parser, OperationState& result) {
  SmallVector<OpAsmParser::OperandType, 4> entryArgs;
  SmallVector<NamedAttrList, 1> argAttrs;
  SmallVector<NamedAttrList, 1> resultAttrs;
  SmallVector<Type, 4> argTypes;
  SmallVector<Type, 1> resultTypes;
  bool isVariadic;

  StringAttr nameAttr;
  // Parse node name.
  if (parser.parseSymbolName(nameAttr, ::mlir::SymbolTable::getSymbolAttrName(),
                             result.attributes))
    return failure();

  auto signatureLocation = parser.getCurrentLocation();
  if (failed(impl::parseFunctionSignature(
          parser, /*allowVariadic=*/false, entryArgs, argTypes, argAttrs,
          isVariadic, resultTypes, resultAttrs)))
    return failure();

  Builder& builder = parser.getBuilder();
  auto type = builder.getFunctionType(argTypes, resultTypes);
  result.addAttribute(NodeOp::getTypeAttrName(), TypeAttr::get(type));

  if (failed(parseAttributions(parser, NodeOp::getNodeIdAttrName(), entryArgs,
                               argTypes)))
    return failure();
  if (failed(parseAttributions(parser, NodeOp::getClusterIdAttrName(),
                               entryArgs, argTypes)))
    return failure();

  // Parse attributes.
  if (failed(parser.parseOptionalAttrDictWithKeyword(result.attributes)))
    return failure();
  mlir::impl::addArgAndResultAttrs(builder, result, argAttrs, resultAttrs);

  // Parse the region. If no argument names were provided, take all names
  // (including those of attributions) from the entry block.
  auto* body = result.addRegion();
  return parser.parseRegion(*body, entryArgs, argTypes);
}

static void printNodeOp(OpAsmPrinter& p, NodeOp op) {
  p << NodeOp::getOperationName() << ' ';
  p.printSymbolName(op.getName());

  FunctionType type = op.getType();
  impl::printFunctionSignature(p, op.getOperation(), type.getInputs(), false,
                               type.getResults());
  // printAttributions(p, op.getNodeIdAttrName(),
  // op.getAttr(op.getNodeIdAttrName()));
  p << ' ' << op.getNodeIdAttrName() << " = "
    << op.getAttrOfType<mlir::IntegerAttr>(op.getNodeIdAttrName());
  p << ' ' << op.getClusterIdAttrName() << " = "
    << op.getAttrOfType<mlir::IntegerAttr>(op.getClusterIdAttrName());
  impl::printFunctionAttributes(p, op.getOperation(), type.getNumInputs(),
                                type.getNumResults(), {});
  p.printRegion(op.getBody(), /*printEntryBlockArgs=*/false);
}

void GraphOp::build(OpBuilder& builder, OperationState& result, StringRef name,
                    ArrayRef<NamedAttribute> attrs) {
  auto funcType = builder.getFunctionType({}, {});

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

void CreateTensorOp::build(OpBuilder& builder, OperationState& result,
                           uint64_t virtualAddress, RankedTensorType type) {
  result.addAttribute(getVirtualAddressAttrName(),
                      builder.getIndexAttr(virtualAddress));
  result.addTypes(type);
}

#define GET_OP_CLASSES
#include "AthenaGraph/AthenaGraphOps.cpp.inc"
} // namespace mlir::ath_graph
