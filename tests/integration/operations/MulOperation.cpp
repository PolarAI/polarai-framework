#include "OperationTest.hpp"

#include <polarai/core/context/Context.hpp>
#include <polarai/core/node/InputNode.hpp>
#include <polarai/core/node/Node.hpp>
#include <polarai/core/node/OutputNode.hpp>
#include <polarai/loaders/MemcpyLoader.hpp>
#include <polarai/operation/MulOperation.hpp>

#include <gtest/gtest.h>

using namespace polarai;
using namespace polarai::core;
using namespace polarai::operation;
using namespace polarai::backend::generic;

template <typename T>
std::vector<T> mulOperation(const std::vector<T>& left,
                            const std::vector<T>& right) {
  std::vector<T> res(left.size());
  for (size_t index = 0; index < left.size(); ++index) {
    res[index] = left[index] * right[index];
  }
  return res;
}

TEST_F(OperationTest, Mul) {
  core::Context context;

  auto graph = context.create<Graph>("mainGraph");

  std::vector<float> left{8, 2, 5, 6};
  std::vector<float> right{3, 2, 1, 0.34};
  auto target = mulOperation(left, right);

  TensorShape shape{2, 2};
  size_t size = shape.getTotalSize();

  auto loader1 =
      context.create<loaders::MemcpyLoader>(left.data(), size * sizeof(float));

  auto loader2 =
      context.create<loaders::MemcpyLoader>(right.data(), size * sizeof(float));

  auto inp1 = graph.create<InputNode>(shape, DataType::FLOAT, false,
                                      loader1.getPublicIndex(), "inp1");
  auto inp2 = graph.create<InputNode>(shape, DataType::FLOAT, false,
                                      loader2.getPublicIndex(), "inp2");

  auto operationId = context.create<MulOperation>();
  auto node = graph.create<Node>(operationId, "logloss");

  graph.connect(inp1, node, MulOperation::LEFT);
  graph.connect(inp2, node, MulOperation::RIGHT);

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
