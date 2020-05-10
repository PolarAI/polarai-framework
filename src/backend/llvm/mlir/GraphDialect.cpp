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

#include "GraphDialect.h"

#include "mlir/IR/Builders.h"
#include "mlir/IR/StandardTypes.h"

#include <algorithm>

// using namespace athena::backend::llvm;
// using namespace ::llvm;

athena::backend::llvm::GraphDialect::GraphDialect(mlir::MLIRContext* ctx)
    : mlir::Dialect("graph", ctx) {
  addOperations<
#define GET_OP_LIST
#include "GraphDialect.cpp.inc"
      >();
}

void athena::backend::llvm::CallOp::build(mlir::OpBuilder& builder,
                                          mlir::OperationState& state,
                                          llvm::StringRef callee,
                                          ArrayRef<mlir::Value> arguments) {
  // Generic call always returns an unranked Tensor initially.
  state.addTypes(UnrankedTensorType::get(builder.getF64Type()));
  state.addOperands(arguments);
  state.addAttribute("callee", builder.getSymbolRefAttr(callee));
}

static mlir::Type athenaToMlirType(athena::core::DataType dataType,
                                   mlir::MLIRContext* ctx) {
  switch (dataType) {
  case athena::core::DataType::FLOAT:
    return mlir::FloatType::get(mlir::StandardTypes::Kind::F32, ctx);
  case athena::core::DataType::DOUBLE:
    return mlir::FloatType::get(mlir::StandardTypes::Kind::F64, ctx);
  case athena::core::DataType::HALF:
    return mlir::FloatType::get(mlir::StandardTypes::Kind::F16, ctx);
  default:
    llvm_unreachable("Unknown data type");
  }
}

void athena::backend::llvm::AllocaOp::build(
    mlir::OpBuilder& builder, OperationState& result,
    const athena::core::inner::Tensor& tensor, int64_t node_id,
    StringRef node_name, int64_t cluster_id) {
  std::vector<int64_t> shape;
  std::transform(tensor.getShapeView().begin(), tensor.getShapeView().end(),
                 std::back_inserter(shape),
                 [&](const size_t item) { return static_cast<int64_t>(item); });

  Type type = athenaToMlirType(tensor.getDataType(), builder.getContext());
  auto resType = RankedTensorType::get(shape, type);
  result.addTypes(resType);
  auto tensorAddrAttr = builder.getI64IntegerAttr(tensor.getVirtualAddress());
  auto nodeIdAttr = builder.getI64IntegerAttr(node_id);
  auto clusterIdAttr = builder.getI64IntegerAttr(cluster_id);
  auto nodeNameAttr = builder.getStringAttr(node_name);

  result.addAttribute("tensor_addr", tensorAddrAttr);
  result.addAttribute("node_id", nodeIdAttr);
  result.addAttribute("node_name", nodeNameAttr);
  result.addAttribute("cluster_id", clusterIdAttr);
}

void athena::backend::llvm::AddOp::build(
    mlir::OpBuilder& builder, OperationState& result, const mlir::Value& a,
    const mlir::Value& b, const athena::core::inner::Tensor& c, int64_t node_id,
    StringRef node_name, int64_t cluster_id) {
  result.addOperands({a, b});

  std::vector<int64_t> shape;
  std::transform(c.getShapeView().begin(), c.getShapeView().end(),
                 std::back_inserter(shape),
                 [&](const size_t item) { return static_cast<int64_t>(item); });

  Type type = athenaToMlirType(c.getDataType(), builder.getContext());
  auto resType = RankedTensorType::get(shape, type);
  result.addTypes(resType);
  auto tensorAddrAttr = builder.getI64IntegerAttr(c.getVirtualAddress());
  auto nodeIdAttr = builder.getI64IntegerAttr(node_id);
  auto clusterIdAttr = builder.getI64IntegerAttr(cluster_id);
  auto nodeNameAttr = builder.getStringAttr(node_name);

  result.addAttribute("tensor_addr", tensorAddrAttr);
  result.addAttribute("node_id", nodeIdAttr);
  result.addAttribute("node_name", nodeNameAttr);
  result.addAttribute("cluster_id", clusterIdAttr);
}

void athena::backend::llvm::MulOp::build(
    mlir::OpBuilder& builder, OperationState& result, const mlir::Value& a,
    const mlir::Value& b, const athena::core::inner::Tensor& c, int64_t node_id,
    StringRef node_name, int64_t cluster_id) {
  result.addOperands({a, b});

  std::vector<int64_t> shape;
  std::transform(c.getShapeView().begin(), c.getShapeView().end(),
                 std::back_inserter(shape),
                 [&](const size_t item) { return static_cast<int64_t>(item); });

  Type type = athenaToMlirType(c.getDataType(), builder.getContext());
  auto resType = RankedTensorType::get(shape, type);
  result.addTypes(resType);
  auto tensorAddrAttr = builder.getI64IntegerAttr(c.getVirtualAddress());
  auto nodeIdAttr = builder.getI64IntegerAttr(node_id);
  auto clusterIdAttr = builder.getI64IntegerAttr(cluster_id);
  auto nodeNameAttr = builder.getStringAttr(node_name);

  result.addAttribute("tensor_addr", tensorAddrAttr);
  result.addAttribute("node_id", nodeIdAttr);
  result.addAttribute("node_name", nodeNameAttr);
  result.addAttribute("cluster_id", clusterIdAttr);
}

void athena::backend::llvm::MatmulOp::build(
    mlir::OpBuilder& builder, OperationState& result, const mlir::Value& a,
    const mlir::Value& b, bool transposeA, bool transposeB, uint64_t alpha,
    uint64_t beta, const athena::core::inner::Tensor& c, int64_t node_id,
    StringRef node_name, int64_t cluster_id) {
  result.addOperands({a, b});

  std::vector<int64_t> shape;
  std::transform(c.getShapeView().begin(), c.getShapeView().end(),
                 std::back_inserter(shape),
                 [&](const size_t item) { return static_cast<int64_t>(item); });

  Type type = athenaToMlirType(c.getDataType(), builder.getContext());
  auto resType = RankedTensorType::get(shape, type);
  result.addTypes(resType);
  auto tensorAddrAttr = builder.getI64IntegerAttr(c.getVirtualAddress());
  auto nodeIdAttr = builder.getI64IntegerAttr(node_id);
  auto clusterIdAttr = builder.getI64IntegerAttr(cluster_id);
  auto nodeNameAttr = builder.getStringAttr(node_name);
  auto trasnpAAttr = builder.getBoolAttr(transposeA);
  auto trasnpBAttr = builder.getBoolAttr(transposeB);
  mlir::Attribute alphaAttr, betaAttr;

  switch (c.getDataType()) {
  case core::DataType::FLOAT:
    alphaAttr = builder.getF32FloatAttr(*reinterpret_cast<float*>(&alpha));
    betaAttr = builder.getF32FloatAttr(*reinterpret_cast<float*>(&beta));
    break;
  case core::DataType::DOUBLE:
    alphaAttr = builder.getF64FloatAttr(*reinterpret_cast<double*>(&alpha));
    betaAttr = builder.getF64FloatAttr(*reinterpret_cast<double*>(&beta));
    break;
  default:
    llvm_unreachable("Unsupported type");
  }

  result.addAttribute("transposeA", trasnpAAttr);
  result.addAttribute("transposeB", trasnpBAttr);
  result.addAttribute("tensor_addr", tensorAddrAttr);
  result.addAttribute("node_id", nodeIdAttr);
  result.addAttribute("node_name", nodeNameAttr);
  result.addAttribute("cluster_id", clusterIdAttr);
}

static mlir::LogicalResult verify(athena::backend::llvm::ReturnOp op) {
  return mlir::success();
}

namespace athena::backend::llvm {
using namespace ::llvm;
#define GET_OP_CLASSES
#include "GraphDialect.cpp.inc"
} // namespace athena::backend::llvm
