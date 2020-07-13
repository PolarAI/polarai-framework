#include "OperationTest.hpp"

#include <polarai/core/context/Context.hpp>
#include <polarai/core/node/InputNode.hpp>
#include <polarai/core/node/Node.hpp>
#include <polarai/core/node/OutputNode.hpp>
#include <polarai/loaders/MemcpyLoader.hpp>
#include <polarai/operation/CombineOperation.hpp>

#include <gtest/gtest.h>

using namespace polarai;
using namespace polarai::core;
using namespace polarai::operation;
using namespace polarai::backend::generic;

template <typename T>
std::vector<T> combineOperation(T alpha, const std::vector<T>& left, T beta,
                                const std::vector<T>& right) {
  std::vector<T> res(left.size());
  for (size_t index = 0; index < left.size(); ++index) {
    res[index] = alpha * left[index] + beta * right[index];
  }
  return res;
}

TEST_F(OperationTest, Combine) {
  core::Context context;

  auto graph = context.create<Graph>("mainGraph");

  std::vector<float> data1{1, 2, 3, 4};
  std::vector<float> data2{7, 8, 9, 10};
  float alpha = 0.34, beta = 0.21;
  auto target = combineOperation(alpha, data1, beta, data2);

  TensorShape shape{2, 2};
  size_t size = shape.getTotalSize();

  auto loader1 =
      context.create<loaders::MemcpyLoader>(data1.data(), size * sizeof(float));

  auto loader2 =
      context.create<loaders::MemcpyLoader>(data2.data(), size * sizeof(float));

  auto inp1 = graph.create<InputNode>(shape, DataType::FLOAT, false,
                                      loader1.getPublicIndex(), "inp1");
  auto inp2 = graph.create<InputNode>(shape, DataType::FLOAT, false,
                                      loader2.getPublicIndex(), "inp2");

  auto operationId = context.create<CombineOperation>(alpha, beta);
  auto node = graph.create<Node>(operationId, "combine");

  graph.connect(inp1, node, CombineOperation::ALPHA);
  graph.connect(inp2, node, CombineOperation::BETA);

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
