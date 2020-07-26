#include "OperationTest.hpp"

#include <polarai/core/context/Context.hpp>
#include <polarai/core/node/InputNode.hpp>
#include <polarai/core/node/Node.hpp>
#include <polarai/core/node/OutputNode.hpp>
#include <polarai/loaders/MemcpyLoader.hpp>
#include <polarai/operation/DivideOperation.hpp>

#include <gtest/gtest.h>

using namespace polarai;
using namespace polarai::core;
using namespace polarai::operation;
using namespace polarai::backend::generic;

template <typename T>
std::vector<T> divideOperation(const std::vector<T>& numerator,
                               const std::vector<T>& denominator) {
  std::vector<T> res(numerator.size());
  for (size_t index = 0; index < numerator.size(); ++index) {
    res[index] = numerator[index] / denominator[index];
  }
  return res;
}

TEST_F(OperationTest, Divide) {
  core::Context context;

  auto graph = context.create<Graph>("mainGraph");

  std::vector<float> data1{1, 2, 3, 4};
  std::vector<float> data2{7, 8, 9, 10};
  auto target = divideOperation(data1, data2);

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

  auto operationId = context.create<DivideOperation>();
  auto node = graph.create<Node>(operationId, "divide");

  graph.connect(inp1, node, DivideOperation::NUMERATOR);
  graph.connect(inp2, node, DivideOperation::DENOMINATOR);

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
