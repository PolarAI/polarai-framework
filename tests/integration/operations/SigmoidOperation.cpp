#include "OperationTest.hpp"

#include <polarai/core/context/Context.hpp>
#include <polarai/core/node/InputNode.hpp>
#include <polarai/core/node/Node.hpp>
#include <polarai/core/node/OutputNode.hpp>
#include <polarai/loaders/MemcpyLoader.hpp>
#include <polarai/operation/SigmoidOperation.hpp>

#include <gtest/gtest.h>

#include <cmath>

using namespace polarai;
using namespace polarai::core;
using namespace polarai::operation;
using namespace polarai::backend::generic;

template <typename T>
std::vector<T> sigmoidOperation(const std::vector<T>& input) {
  constexpr T eps = 1e-5;
  std::vector<T> res(input.size());
  for (size_t index = 0; index < input.size(); ++index) {
    res[index] = 1 / (1 + exp(-input[index]));
    if (std::abs(res[index]-1) < eps) {
      res[index] = 1 - eps;
    } else if (fabs(res[index]) < eps) {
      res[index] = eps;
    }
  }
  return res;
}

TEST_F(OperationTest, DISABLED_Sigmoid) {
  core::Context context;

  auto graph = context.create<Graph>("mainGraph");

  std::vector<float> input{-100, 2, 5, 50};
  auto target = sigmoidOperation(input);

  TensorShape shape{2, 2};
  size_t size = shape.getTotalSize();

  auto loader =
      context.create<loaders::MemcpyLoader>(input.data(), size * sizeof(float));

  auto inp = graph.create<InputNode>(shape, DataType::FLOAT, false,
                                     loader.getPublicIndex(), "inp");

  auto operationId = context.create<SigmoidOperation>();
  auto node = graph.create<Node>(operationId, "sigmoid");

  graph.connect(inp, node, SigmoidOperation::Unmarked);

  auto out = graph.create<OutputNode>("out");
  graph.connect(node, out, Operation::Unmarked);

  withEachDeviceDo([&graph, out, &context, &target](GenericExecutor& executor) {
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
