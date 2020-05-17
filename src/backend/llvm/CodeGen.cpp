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

#include <AthenaGraph/AthenaGraphOps.h>
#include <athena/backend/llvm/CodeGen.h>
#include <athena/core/Allocator.h>
#include <athena/core/DataType.h>
#include <athena/core/Generator.h>
#include <athena/core/inner/Tensor.h>

#include <bits/stdint-intn.h>
#include <cstddef>
#include <llvm/ADT/APFloat.h>
#include <llvm/ADT/SmallVector.h>
#include <llvm/ADT/StringRef.h>
#include <mlir/Dialect/StandardOps/IR/Ops.h>
#include <mlir/IR/Attributes.h>
#include <mlir/IR/Builders.h>
#include <mlir/IR/StandardTypes.h>
#include <mlir/IR/Types.h>
#include <mlir/IR/Value.h>

#include <any>
#include <functional>
#include <string_view>
#include <system_error>
#include <variant>
#include <vector>

using namespace athena::core;

struct MlirNode : public GenNode {
  using GenNode::GenNode;

  MlirNode(const MlirNode&) = default;
  MlirNode(MlirNode&&) = default;
  MlirNode& operator=(const MlirNode&) = default;
  MlirNode& operator=(MlirNode&&) = default;

  auto getOperand(size_t idx) -> GenValue override {
    auto mlirNode = std::any_cast<mlir::ath_graph::NodeOp>(node);
    return GenValue{mlirNode.getArgument(idx)};
  }

  auto getResult() -> GenValue override {
    auto mlirNode = std::any_cast<mlir::ath_graph::NodeOp>(node);
    mlir::Value res = mlirNode.getBody().front().getTerminator()->getOperand(0);
    return GenValue{res};
  }

  auto getBatchIndex() -> GenValue override {
    auto mlirNode = std::any_cast<mlir::ath_graph::NodeOp>(node);
    return GenValue{mlirNode.getBatchIndex()};
  }
};

static auto getTensorType(mlir::OpBuilder& builder, const inner::Tensor& tensor)
    -> mlir::Type {
  ::llvm::SmallVector<int64_t, 3> shape;
  for (auto dim : tensor.getShapeView()) {
    shape.push_back(dim);
  }
  mlir::Type dataType;

  if (tensor.getDataType() == DataType::FLOAT) {
    dataType = builder.getF32Type();
  } else if (tensor.getDataType() == DataType::DOUBLE) {
    dataType = builder.getF64Type();
  } else if (tensor.getDataType() == DataType::HALF) {
    dataType = builder.getF16Type();
  }

  return mlir::RankedTensorType::get(shape, dataType);
}

namespace athena::backend::llvm {
void populateCodeGenPatterns(athena::core::Generator& generator,
                             mlir::OpBuilder& builder) {

  //===--------------------------------------------------------------------===//
  // Utility functors
  //===--------------------------------------------------------------------===//

  std::function<GenValue(Generator::SupportedConstantT)> constantFunctor =
      [&](Generator::SupportedConstantT constant) {
        mlir::Value resultValue;

        if (std::holds_alternative<int32_t>(constant)) {
          resultValue =
              builder
                  .create<mlir::ConstantIntOp>(builder.getUnknownLoc(),
                                               std::get<int32_t>(constant), 32)
                  .getResult();
        } else if (std::holds_alternative<uint32_t>(constant)) {
          resultValue =
              builder
                  .create<mlir::ConstantIntOp>(builder.getUnknownLoc(),
                                               std::get<uint32_t>(constant), 32)
                  .getResult();
        } else if (std::holds_alternative<int64_t>(constant)) {
          resultValue =
              builder
                  .create<mlir::ConstantIntOp>(builder.getUnknownLoc(),
                                               std::get<int64_t>(constant), 64)
                  .getResult();
        } else if (std::holds_alternative<uint64_t>(constant)) {
          resultValue =
              builder
                  .create<mlir::ConstantIntOp>(builder.getUnknownLoc(),
                                               std::get<uint64_t>(constant), 64)
                  .getResult();
        } else if (std::holds_alternative<float>(constant)) {
          ::llvm::APFloat val(std::get<float>(constant));
          auto type = builder.getF32Type();
          resultValue = builder
                            .create<mlir::ConstantFloatOp>(
                                builder.getUnknownLoc(), val, type)
                            .getResult();
        } else if (std::holds_alternative<double>(constant)) {
          ::llvm::APFloat val(std::get<double>(constant));
          auto type = builder.getF64Type();
          resultValue = builder
                            .create<mlir::ConstantFloatOp>(
                                builder.getUnknownLoc(), val, type)
                            .getResult();
        }

        return GenValue{resultValue};
      };
  generator.registerConstantFunctor(constantFunctor);

  std::function<GenNode(std::string_view, size_t, size_t,
                        const std::vector<inner::Tensor>&, inner::Tensor&)>
      nodeFunctor = [&](std::string_view name, size_t nodeId, size_t clusterId,
                        const std::vector<inner::Tensor>& operands,
                        inner::Tensor& out) {
        ::llvm::SmallVector<mlir::Type, 5> nodeOperandTypes;

        for (const auto& tensor : operands) {
          auto tensorType = getTensorType(builder, tensor);
          nodeOperandTypes.push_back(tensorType);
        }

        mlir::FunctionType nodeType;

        if (out.getVirtualAddress() == 0) {
          nodeType = builder.getFunctionType(nodeOperandTypes, {});
        } else {
          nodeType = builder.getFunctionType(nodeOperandTypes,
                                             {getTensorType(builder, out)});
        }

        auto node = builder.create<mlir::ath_graph::NodeOp>(
            builder.getUnknownLoc(), name, nodeType, nodeId, clusterId);

        {
          mlir::OpBuilder::InsertionGuard guard(builder);
          builder.setInsertionPointToStart(&node.getBody().front());

          if (out.getVirtualAddress() == 0) {
            builder.create<mlir::ath_graph::ReturnOp>(builder.getUnknownLoc());
          } else {
            auto context = node.getContext();
            auto res = builder.create<mlir::ath_graph::GetTensor>(
                builder.getUnknownLoc(), context, out.getVirtualAddress(),
                getTensorType(builder, out).cast<mlir::RankedTensorType>());
            builder.create<mlir::ath_graph::ReturnOp>(builder.getUnknownLoc(),
                                                      res.getResult());
          }
        }

        return MlirNode{node};
      };
  generator.registerNodeFunctor(nodeFunctor);

  std::function<GenGraph(std::string_view, size_t)> graphFunctor =
      [&](std::string_view name, size_t graphId) {
        // fixme set graph ID
        auto graph = builder.create<mlir::ath_graph::GraphOp>(
            builder.getUnknownLoc(), name);
        return GenGraph{graph};
      };
  generator.registerGraphFunctor(graphFunctor);

  auto setInsertionPointFunctor = [&](GenInsertionPoint insertionPoint) {
    auto mlirPoint =
        std::any_cast<mlir::OpBuilder::InsertPoint>(insertionPoint.point);
    builder.restoreInsertionPoint(mlirPoint);
  };
  generator.registerSetInsertionPointFunctor(setInsertionPointFunctor);

  auto setNodeInsertionPointFunctor = [&](GenNode node) {
    auto mlirNode = std::any_cast<mlir::ath_graph::NodeOp>(node.node);
    auto it = mlirNode.getBody().front().without_terminator().end();
    it--;

    builder.setInsertionPointAfter(&*it);
  };
  generator.registerSetInsertionPointFunctor(setNodeInsertionPointFunctor);


  auto setGraphInsertionPointFunctor = [&](GenGraph graph) {
    auto mlirNode = std::any_cast<mlir::ath_graph::GraphOp>(graph.graph);
    auto it = mlirNode.body().front().without_terminator().end();
    it--;

    builder.setInsertionPointAfter(&*it);
  };
  generator.registerSetInsertionPointFunctor(setGraphInsertionPointFunctor);

  auto getInsertionPointFunctor = [&]() {
    return GenInsertionPoint{builder.saveInsertionPoint()};
  };
  generator.registerGetInsertionPointFunctor(getInsertionPointFunctor);

  //===--------------------------------------------------------------------===//
  // Builtin functors
  //===--------------------------------------------------------------------===//

  std::function<GenValue(GenValue)> allocFunctor =
      [&](GenValue tensor) -> GenValue {
    auto tensorVal = std::any_cast<mlir::Value>(tensor.value);
    builder.create<mlir::ath_graph::AllocOp>(builder.getUnknownLoc(),
                                             tensorVal);

    return GenValue{mlir::Value()};
  };
  generator.registerFunctor<GenValue>(builtin::Alloc, allocFunctor);

  std::function<GenValue(GenValue, core::LockType)> lockFunctor =
      [&](GenValue tensor, core::LockType lockType) -> GenValue {
    auto tensorVal = std::any_cast<mlir::Value>(tensor.value);
    mlir::StringAttr mlirLockType;
    if (lockType == core::LockType::READ) {
      mlirLockType = builder.getStringAttr("read");
    } else {
      mlirLockType = builder.getStringAttr("read_write");
    }

    builder.create<mlir::ath_graph::LockOp>(builder.getUnknownLoc(),
                                            mlirLockType, tensorVal);
    return GenValue{mlir::Value()};
  };

  generator.registerFunctor(builtin::Lock, lockFunctor);

  std::function<GenValue(GenValue)> releaseFunctor = [&](GenValue tensor) {
    auto tensorVal = std::any_cast<mlir::Value>(tensor.value);

    builder.create<mlir::ath_graph::ReleaseOp>(builder.getUnknownLoc(),
                                               tensorVal);

    return GenValue{mlir::Value()};
  };
  generator.registerFunctor(builtin::Release, releaseFunctor);

  // todo barrier

  std::function<GenValue(std::string_view, GenValue)> invokeLoaderFunctor =
      [&](std::string_view loaderRoutine, GenValue destTensor) {
        auto tensorVal = std::any_cast<mlir::Value>(destTensor.value);

        builder.create<mlir::ath_graph::InvokeLoaderOp>(
            builder.getUnknownLoc(), builder.getStringAttr(loaderRoutine),
            tensorVal);

        return GenValue{mlir::Value()};
      };
  generator.registerFunctor(builtin::InvokeLoader, invokeLoaderFunctor);

  std::function<GenValue(GenGraph, GenNode, const std::vector<GenValue>&)>
      evalFunctor = [&](GenGraph graph, GenNode node,
                        const std::vector<GenValue>& operands) {
        auto mlirGraph = std::any_cast<mlir::ath_graph::GraphOp>(graph.graph);
        auto mlirNode = std::any_cast<mlir::ath_graph::NodeOp>(node.node);

        ::llvm::SmallVector<mlir::Value, 8> nodeOperands;
        for (auto& op : operands) {
          auto val = std::any_cast<mlir::Value>(op.value);
          nodeOperands.push_back(val);
        }
        nodeOperands.push_back(mlirGraph.getContext());
        nodeOperands.push_back(mlirGraph.getBatchSize());

        auto evalRes = builder.create<mlir::ath_graph::EvalOp>(
            builder.getUnknownLoc(), mlirNode, nodeOperands);

        if (evalRes.getNumResults() == 1) {
          return GenValue{evalRes.getResult(0)};
        }
        return GenValue{mlir::Value()};
      };
  generator.registerFunctor(builtin::NodeEval, evalFunctor);

  std::function<GenValue(GenValue, GenValue, GenValue, GenValue, GenValue)>
      addFunctor = [&](GenValue a, GenValue scaleA, GenValue b, GenValue scaleB,
                       GenValue out) {
        auto aVal = std::any_cast<mlir::Value>(a.value);
        auto scaleAVal = std::any_cast<mlir::Value>(scaleA.value);
        auto bVal = std::any_cast<mlir::Value>(b.value);
        auto scaleBVal = std::any_cast<mlir::Value>(scaleB.value);
        auto outVal = std::any_cast<mlir::Value>(out.value);

        builder.create<mlir::ath_graph::AddOp>(
            builder.getUnknownLoc(), aVal, scaleAVal, bVal, scaleBVal, outVal);

        return GenValue{};
      };
  generator.registerFunctor(builtin::Add, addFunctor);

  std::function<GenValue(GenValue, GenValue, GenValue, GenValue)> mulFunctor =
      [&](GenValue a, GenValue b, GenValue scale, GenValue out) {
        auto aVal = std::any_cast<mlir::Value>(a.value);
        auto bVal = std::any_cast<mlir::Value>(b.value);
        auto scaleVal = std::any_cast<mlir::Value>(scale.value);
        auto outVal = std::any_cast<mlir::Value>(out.value);

        builder.create<mlir::ath_graph::MulOp>(builder.getUnknownLoc(), aVal,
                                               bVal, scaleVal, outVal);
        return GenValue{};
      };
  generator.registerFunctor(builtin::Mul, mulFunctor);

  std::function<GenValue(GenValue, GenValue, GenValue, GenValue, GenValue)>
      matmulFunctor = [&](GenValue a, GenValue scaleA, GenValue b,
                          GenValue scaleB, GenValue out) {
        auto aVal = std::any_cast<mlir::Value>(a.value);
        auto scaleAVal = std::any_cast<mlir::Value>(scaleA.value);
        auto bVal = std::any_cast<mlir::Value>(b.value);
        auto scaleBVal = std::any_cast<mlir::Value>(scaleB.value);
        auto outVal = std::any_cast<mlir::Value>(out.value);

        builder.create<mlir::ath_graph::MatmulOp>(
            builder.getUnknownLoc(), outVal.getType(), aVal, scaleAVal, bVal,
            scaleBVal, outVal);

        return GenValue{};
      };
  generator.registerFunctor(builtin::MatMul, matmulFunctor);

  std::function<GenValue(GenValue, GenValue)> fillFunctor =
      [&](GenValue pattern, GenValue out) {
        auto patternVal = std::any_cast<mlir::Value>(pattern.value);
        auto outVal = std::any_cast<mlir::Value>(out.value);

        builder.create<mlir::ath_graph::FillOp>(builder.getUnknownLoc(),
                                                patternVal, outVal);

        return GenValue{};
      };
  generator.registerFunctor(builtin::Fill, fillFunctor);

  std::function<GenValue(GenValue, GenValue)> sliceFunctor =
      [&](GenValue index, GenValue tensor) {
        auto indexVal = std::any_cast<mlir::Value>(index.value);
        auto outVal = std::any_cast<mlir::Value>(tensor.value);

        auto res = builder.create<mlir::ath_graph::SliceOp>(
            builder.getUnknownLoc(), indexVal, outVal);
        return GenValue{res.getResult()};
      };
  generator.registerFunctor(builtin::Slice, sliceFunctor);

  std::function<GenValue(GenValue, GenValue)> transposeFunctor =
      [&](GenValue tensor, GenValue out) {
        auto tensorVal = std::any_cast<mlir::Value>(tensor.value);
        auto outVal = std::any_cast<mlir::Value>(out.value);

        builder.create<mlir::ath_graph::TransposeOp>(builder.getUnknownLoc(),
                                                     tensorVal, outVal);
        return GenValue{};
      };
  generator.registerFunctor(builtin::Transpose, transposeFunctor);
}
} // namespace athena::backend::llvm