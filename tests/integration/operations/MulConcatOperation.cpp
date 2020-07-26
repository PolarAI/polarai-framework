#include "OperationTest.hpp"

#include <polarai/core/context/Context.hpp>
#include <polarai/core/node/InputNode.hpp>
#include <polarai/core/node/Node.hpp>
#include <polarai/core/node/OutputNode.hpp>
#include <polarai/loaders/MemcpyLoader.hpp>
#include <polarai/operation/MulConcatOperation.hpp>

#include <gtest/gtest.h>

using namespace polarai;
using namespace polarai::core;
using namespace polarai::operation;
using namespace polarai::backend::generic;

template <typename T>
std::vector<T> mulConcatOperation(const std::vector<T>& localDerivative,
                                  const std::vector<T>& gradient) {
  std::vector<T> res(localDerivative.size());
  for (size_t index = 0; index < localDerivative.size(); ++index) {
    res[index] = localDerivative[index] * gradient[0];
  }
  return res;
}

TEST_F(OperationTest, MulConcat) {
  core::Context context;

  auto graph = context.create<Graph>("mainGraph");

  std::vector<float> localDerivative{-100, 2, 5, 50};
  std::vector<float> gradient{17.5};
  auto target = mulConcatOperation(localDerivative, gradient);

  TensorShape shape{2, 2};
  size_t size = shape.getTotalSize();

  auto loaderLocalDerivative = context.create<loaders::MemcpyLoader>(
      localDerivative.data(), size * sizeof(float));
  auto inpLocalDerivative = graph.create<InputNode>(
      shape, DataType::FLOAT, false, loaderLocalDerivative.getPublicIndex(),
      "inpLocDeriv");

  auto loaderGradient =
      context.create<loaders::MemcpyLoader>(gradient.data(), 1 * sizeof(float));
  auto inpGradient =
      graph.create<InputNode>(TensorShape{1, 1}, DataType::FLOAT, false,
                              loaderGradient.getPublicIndex(), "inpGradient");

  auto operationId = context.create<MulConcatOperation>();
  auto node = graph.create<Node>(operationId, "mulconcat");

  graph.connect(inpLocalDerivative, node, MulConcatOperation::LOCAL_DERIVATIVE);
  graph.connect(inpGradient, node, MulConcatOperation::GRADIENT);

  auto out = graph.create<OutputNode>("out");
  graph.connect(node, out, Operation::Unmarked);

  withEachDeviceDo([&graph, out, &context, &target](Executor& executor) {
    executor.addGraph(graph);
    executor.evaluate(graph);

    auto accessor =
        context.internal()->getRef<OutputNode>(out).getAccess<float>(
            executor.getAllocator());
    for (int i = 0; i < 4; i++) {
      EXPECT_FLOAT_EQ(accessor(i), target[i]);
    }
  });
}
