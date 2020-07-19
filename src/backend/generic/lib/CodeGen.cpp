//===----------------------------------------------------------------------===//
// Copyright (c) 2020 Polar. All rights reserved.
//
// Licensed under MIT license.
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS, WITHOUT
// WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the
// License for the specific language governing permissions and limitations under
// the License.
//===----------------------------------------------------------------------===//

#include <PolarGraph/PolarGraphOps.h>
#include <polarai/backend/generic/CodeGen.hpp>
#include <polarai/core/Generator.hpp>
#include <polarai/core/internal/GenBuiltins.hpp>
#include <polarai/core/tensor/DataType.hpp>
#include <polarai/core/tensor/internal/TensorInternal.hpp>

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
#include <memory>
#include <string_view>
#include <system_error>
#include <variant>
#include <vector>

using namespace polarai::core;
using namespace polarai::core::internal;

struct MlirValueImpl : internal::GenValueImplBase {
  mlir::Value value;
  MlirValueImpl(mlir::Value val) : value(val) {}
};

struct MlirNodeImpl : internal::GenNodeImplBase {
  mlir::polar_graph::NodeOp node;
  MlirNodeImpl(mlir::polar_graph::NodeOp node) : node(node) {}
  auto getOperand(size_t i) -> GenValue override {
    mlir::Value op = node.getArgument(i);
    return GenValue(std::make_shared<MlirValueImpl>(op));
  };
  auto getResult() -> GenValue override {
    mlir::Value op = node.getBody().front().front().getResult(0);
    return GenValue(std::make_shared<MlirValueImpl>(op));
  };
  auto getBatchIndex() -> GenValue override {
    mlir::Value op = node.getBody().front().getTerminator()->getOperand(0);
    return GenValue(std::make_shared<MlirValueImpl>(op));
  };
};

struct MlirGraphImpl : internal::GenGraphImplBase {
  mlir::polar_graph::GraphOp graph;
  MlirGraphImpl(mlir::polar_graph::GraphOp graph) : graph(graph) {}
};

struct MlirInsPointImpl : internal::GenInsPointImplBase {
  mlir::OpBuilder::InsertPoint point;
  MlirInsPointImpl(mlir::OpBuilder::InsertPoint ip) : point(ip) {}
};

static auto getTensorType(mlir::OpBuilder& builder,
                          const internal::TensorInternal& tensor)
    -> mlir::RankedTensorType {
  ::llvm::SmallVector<int64_t, 3> shape;
  for (long dim : tensor.getShapeView()) {
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

namespace polarai::backend::generic {
void populateCodeGenPatterns(polarai::core::internal::Generator& generator,
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

        return GenValue{std::make_shared<MlirValueImpl>(resultValue)};
      };
  generator.registerConstantFunctor(constantFunctor);

  std::function<GenNode(std::string_view, size_t, size_t,
                        const std::vector<internal::TensorInternal*>&,
                        internal::TensorInternal&)>
      nodeFunctor = [&](std::string_view name, size_t nodeId, size_t clusterId,
                        const std::vector<internal::TensorInternal*>& operands,
                        internal::TensorInternal& out) {
        ::llvm::SmallVector<mlir::Type, 5> nodeOperandTypes;

        for (const auto& tensor : operands) {
          auto tensorType = getTensorType(builder, *tensor);
          nodeOperandTypes.push_back(tensorType);
        }

        mlir::FunctionType nodeType;

        if (out.getVirtualAddress() == 0) {
          nodeType = builder.getFunctionType(nodeOperandTypes, {});
        } else {
          nodeType = builder.getFunctionType(nodeOperandTypes,
                                             {getTensorType(builder, out)});
        }

        auto node = builder.create<mlir::polar_graph::NodeOp>(
            builder.getUnknownLoc(), name, nodeType, nodeId, clusterId);
        mlir::OpBuilder::InsertionGuard guard{builder};
        builder.setInsertionPointToStart(&node.getBody().front());

        if (out.getVirtualAddress() != 0) {
          auto tensorType = getTensorType(builder, out);
          builder.create<mlir::polar_graph::CreateTensorOp>(
              builder.getUnknownLoc(), out.getVirtualAddress(), tensorType);
        }

        return GenNode{std::make_shared<MlirNodeImpl>(node)};
      };
  generator.registerNodeFunctor(nodeFunctor);

  std::function<GenGraph(std::string_view, size_t)> graphFunctor =
      [&](std::string_view name, size_t graphId) {
        // fixme set graph ID
        auto graph = builder.create<mlir::polar_graph::GraphOp>(
            builder.getUnknownLoc(), name);
        return GenGraph{std::make_shared<MlirGraphImpl>(graph)};
      };
  generator.registerGraphFunctor(graphFunctor);

  auto setInsertionPointFunctor = [&](GenInsertionPoint insertionPoint) {
    auto mlirPoint = insertionPoint.point<MlirInsPointImpl>().point;
    builder.restoreInsertionPoint(mlirPoint);
  };
  generator.registerSetInsertionPointFunctor(setInsertionPointFunctor);

  auto setNodeInsertionPointFunctor = [&](GenNode node) {
    auto mlirNode = node.node<MlirNodeImpl>().node;
    auto end = mlirNode.getBody().front().without_terminator().end();
    auto begin = mlirNode.getBody().front().without_terminator().begin();

    builder.setInsertionPointToEnd(&mlirNode.getBody().front());
  };
  generator.registerSetInsertionPointFunctor(setNodeInsertionPointFunctor);

  auto setGraphInsertionPointFunctor = [&](GenGraph graph) {
    auto mlirGraph = graph.graph<MlirGraphImpl>().graph;
    auto end = mlirGraph.body().front().without_terminator().end();
    auto begin = mlirGraph.body().front().without_terminator().begin();
    if (end == begin) {
      builder.setInsertionPointToStart(&mlirGraph.body().front());
    } else {
      end--;
      builder.setInsertionPointAfter(&*end);
    }
  };
  generator.registerSetInsertionPointFunctor(setGraphInsertionPointFunctor);

  auto getInsertionPointFunctor = [&]() {
    return GenInsertionPoint{
        std::make_shared<MlirInsPointImpl>(builder.saveInsertionPoint())};
  };
  generator.registerGetInsertionPointFunctor(getInsertionPointFunctor);

  //===--------------------------------------------------------------------===//
  // Builtin functors
  //===--------------------------------------------------------------------===//

  builtin_functor_t<builtin::Alloc> allocFunctor =
      [&](GenValue tensor) -> GenValue {
    auto tensorVal = tensor.value<MlirValueImpl>().value;
    builder.create<mlir::polar_graph::AllocOp>(builder.getUnknownLoc(),
                                               tensorVal);

    return GenValue{nullptr};
  };
  generator.registerFunctor<builtin::Alloc>(allocFunctor);

  builtin_functor_t<builtin::Lock> lockFunctor =
      [&](GenValue tensor, core::internal::LockType lockType) -> GenValue {
    auto tensorVal = tensor.value<MlirValueImpl>().value;
    mlir::StringAttr mlirLockType;
    if (lockType == core::internal::LockType::READ) {
      mlirLockType = builder.getStringAttr("read");
    } else {
      mlirLockType = builder.getStringAttr("read_write");
    }

    builder.create<mlir::polar_graph::LockOp>(builder.getUnknownLoc(),
                                              mlirLockType, tensorVal);
    return GenValue{nullptr};
  };
  generator.registerFunctor<builtin::Lock>(lockFunctor);

  builtin_functor_t<builtin::Release> releaseFunctor = [&](GenValue tensor) {
    auto tensorVal = tensor.value<MlirValueImpl>().value;

    builder.create<mlir::polar_graph::ReleaseOp>(builder.getUnknownLoc(),
                                                 tensorVal);

    return GenValue{nullptr};
  };
  generator.registerFunctor<builtin::Release>(releaseFunctor);

  builtin_functor_t<builtin::Barrier> barrierBuiltin = [&](uint64_t clusterId) {
    builder.create<mlir::polar_graph::BarrierOp>(
        builder.getUnknownLoc(), builder.getIndexAttr(clusterId));
  };
  generator.registerFunctor<builtin::Barrier>(barrierBuiltin);

  builtin_functor_t<builtin::InvokeLoader> invokeLoaderFunctor =
      [&](GenValue destTensor) {
        auto tensorVal = destTensor.value<MlirValueImpl>().value;

        builder.create<mlir::polar_graph::InvokeLoaderOp>(
            builder.getUnknownLoc(), tensorVal);

        return GenValue{nullptr};
      };
  generator.registerFunctor<builtin::InvokeLoader>(invokeLoaderFunctor);

  builtin_functor_t<builtin::NodeEval> evalFunctor =
      [&](GenGraph graph, GenNode node, const std::vector<GenValue>& operands) {
        auto mlirGraph = graph.graph<MlirGraphImpl>().graph;
        auto mlirNode = node.node<MlirNodeImpl>().node;

        ::llvm::SmallVector<mlir::Value, 8> nodeOperands;
        for (const GenValue& op : operands) {
          auto val = op.value<MlirValueImpl>().value;
          nodeOperands.push_back(val);
        }

        auto evalRes = builder.create<mlir::polar_graph::EvalOp>(
            builder.getUnknownLoc(), mlirNode, nodeOperands);

        if (evalRes.getNumResults() == 1) {
          return GenValue{
              std::make_shared<MlirValueImpl>(evalRes.getResult(0))};
        }
        return GenValue{nullptr};
      };
  generator.registerFunctor<builtin::NodeEval>(evalFunctor);

  builtin_functor_t<builtin::Return> retFunctor =
      [&](std::optional<GenValue> out) {
        if (out) {
          auto outVal = out.value().value<MlirValueImpl>().value;
          builder.create<mlir::polar_graph::ReturnOp>(builder.getUnknownLoc(),
                                                      outVal);
        } else {
          builder.create<mlir::polar_graph::ReturnOp>(builder.getUnknownLoc());
        }
      };
  generator.registerFunctor<builtin::Return>(retFunctor);

  builtin_functor_t<builtin::Add> addFunctor = [&](GenValue a, GenValue scaleA,
                                                   GenValue b, GenValue scaleB,
                                                   GenValue out) {
    auto aVal = a.value<MlirValueImpl>().value;
    auto scaleAVal = scaleA.value<MlirValueImpl>().value;
    auto bVal = b.value<MlirValueImpl>().value;
    auto scaleBVal = scaleB.value<MlirValueImpl>().value;
    auto outVal = out.value<MlirValueImpl>().value;

    builder.create<mlir::polar_graph::AddOp>(
        builder.getUnknownLoc(), aVal, scaleAVal, bVal, scaleBVal, outVal);
  };
  generator.registerFunctor<builtin::Add>(addFunctor);

  builtin_functor_t<builtin::Copy> copyFunctor = [&](GenValue input,
                                                     GenValue out) {
    auto inputVal = input.value<MlirValueImpl>().value;
    auto outVal = out.value<MlirValueImpl>().value;

    builder.create<mlir::polar_graph::CopyOp>(builder.getUnknownLoc(), inputVal,
                                              outVal);
  };
  generator.registerFunctor<builtin::Copy>(copyFunctor);

  builtin_functor_t<builtin::Divide> divideFunctor =
      [&](GenValue numerator, GenValue denominator, GenValue out) {
        auto numeratorVal = numerator.value<MlirValueImpl>().value;
        auto denominatorVal = denominator.value<MlirValueImpl>().value;
        auto outVal = out.value<MlirValueImpl>().value;

        builder.create<mlir::polar_graph::DivideOp>(
            builder.getUnknownLoc(), numeratorVal, denominatorVal, outVal);
      };
  generator.registerFunctor<builtin::Divide>(divideFunctor);

  builtin_functor_t<builtin::LogLoss> logLossFunctor =
      [&](GenValue predicted, GenValue groundTruth, GenValue out) {
        auto predictedVal = predicted.value<MlirValueImpl>().value;
        auto groundTruthVal = groundTruth.value<MlirValueImpl>().value;
        auto outVal = out.value<MlirValueImpl>().value;

        builder.create<mlir::polar_graph::LogLossOp>(
            builder.getUnknownLoc(), predictedVal, groundTruthVal, outVal);
      };
  generator.registerFunctor<builtin::LogLoss>(logLossFunctor);

  builtin_functor_t<builtin::MatMul> matmulFunctor =
      [&](GenValue left, GenValue right, GenValue out, bool transpLeft,
          bool transpRight) {
        auto leftVal = left.value<MlirValueImpl>().value;
        auto rightVal = right.value<MlirValueImpl>().value;
        auto outVal = out.value<MlirValueImpl>().value;

        builder.create<mlir::polar_graph::MatMulOp>(builder.getUnknownLoc(),
                                                    leftVal, rightVal, outVal,
                                                    transpLeft, transpRight);
      };
  generator.registerFunctor<builtin::MatMul>(matmulFunctor);

  builtin_functor_t<builtin::Mul> mulFunctor = [&](GenValue a, GenValue b,
                                                   GenValue out) {
    auto aVal = a.value<MlirValueImpl>().value;
    auto bVal = b.value<MlirValueImpl>().value;
    auto outVal = out.value<MlirValueImpl>().value;

    builder.create<mlir::polar_graph::MulOp>(builder.getUnknownLoc(), aVal,
                                             bVal, outVal);
  };
  generator.registerFunctor<builtin::Mul>(mulFunctor);

  builtin_functor_t<builtin::MulConcat> mulConcatFunctor =
      [&](GenValue gradient, GenValue localDerivative, GenValue out) {
        auto gradientVal = gradient.value<MlirValueImpl>().value;
        auto localDerivativeVal = localDerivative.value<MlirValueImpl>().value;
        auto outVal = out.value<MlirValueImpl>().value;

        builder.create<mlir::polar_graph::MulConcatOp>(
            builder.getUnknownLoc(), gradientVal, localDerivativeVal, outVal);
      };
  generator.registerFunctor<builtin::MulConcat>(mulConcatFunctor);

  builtin_functor_t<builtin::ReLU> reluFunctor = [&](GenValue input,
                                                           GenValue out) {
    auto inputVal = input.value<MlirValueImpl>().value;
    auto outVal = out.value<MlirValueImpl>().value;

    builder.create<mlir::polar_graph::ReLUOp>(builder.getUnknownLoc(),
                                                 inputVal, outVal);
  };
  generator.registerFunctor<builtin::ReLU>(reluFunctor);

  builtin_functor_t<builtin::Sigmoid> sigmoidFunctor = [&](GenValue input,
                                                           GenValue out) {
    auto inputVal = input.value<MlirValueImpl>().value;
    auto outVal = out.value<MlirValueImpl>().value;

    builder.create<mlir::polar_graph::SigmoidOp>(builder.getUnknownLoc(),
                                                 inputVal, outVal);
  };
  generator.registerFunctor<builtin::Sigmoid>(sigmoidFunctor);

  builtin_functor_t<builtin::Softmax> softmaxFunctor = [&](GenValue input,
                                                           GenValue out) {
    auto inputVal = input.value<MlirValueImpl>().value;
    auto outVal = out.value<MlirValueImpl>().value;

    builder.create<mlir::polar_graph::SoftmaxOp>(builder.getUnknownLoc(),
                                                 inputVal, outVal);
  };
  generator.registerFunctor<builtin::Softmax>(softmaxFunctor);

  builtin_functor_t<builtin::Fill> fillFunctor = [&](GenValue pattern,
                                                     GenValue out) {
    auto patternVal = pattern.value<MlirValueImpl>().value;
    auto outVal = out.value<MlirValueImpl>().value;

    builder.create<mlir::polar_graph::FillOp>(builder.getUnknownLoc(),
                                              patternVal, outVal);
  };
  generator.registerFunctor<builtin::Fill>(fillFunctor);

  builtin_functor_t<builtin::Slice> sliceFunctor = [&](GenValue index,
                                                       GenValue tensor) {
    auto indexVal = index.value<MlirValueImpl>().value;
    auto outVal = tensor.value<MlirValueImpl>().value;

    builder.create<mlir::polar_graph::SliceOp>(builder.getUnknownLoc(),
                                               indexVal, outVal);
  };
  generator.registerFunctor<builtin::Slice>(sliceFunctor);

  builtin_functor_t<builtin::Transpose> transposeFunctor = [&](GenValue tensor,
                                                               GenValue out) {
    auto tensorVal = tensor.value<MlirValueImpl>().value;
    auto outVal = out.value<MlirValueImpl>().value;

    builder.create<mlir::polar_graph::TransposeOp>(builder.getUnknownLoc(),
                                                   tensorVal, outVal);
  };
  generator.registerFunctor<builtin::Transpose>(transposeFunctor);
}
} // namespace polarai::backend::llvm
