#include "OperationTest.hpp"

#include <polarai/core/context/Context.hpp>
#include <polarai/core/node/InputNode.hpp>
#include <polarai/core/node/Node.hpp>
#include <polarai/core/node/OutputNode.hpp>
#include <polarai/loaders/MemcpyLoader.hpp>
#include <polarai/operation/AddOperation.hpp>

#include <gtest/gtest.h>

using namespace polarai;
using namespace polarai::core;
using namespace polarai::operation;
using namespace polarai::backend::generic;

TEST_F(OperationTest, AddOp1D) {
  core::Context context;

  auto graph = context.create<Graph>("mainGraph");

  std::vector<float> data1{1, 2, 3, 4};
  std::vector<float> data2{7, 8, 9, 10};
  std::vector<float> target{8, 10, 12, 14};

  TensorShape shape{4};
  size_t size = shape.getTotalSize();

  auto loader1 = context.create<loaders::MemcpyLoader>(
      data1.data(), data1.size() * sizeof(float));

  auto loader2 = context.create<loaders::MemcpyLoader>(
      data2.data(), data2.size() * sizeof(float));

  auto inp1 = graph.create<InputNode>(shape, DataType::FLOAT, false,
                                      loader1.getPublicIndex(), "inp1");
  auto inp2 = graph.create<InputNode>(shape, DataType::FLOAT, false,
                                      loader2.getPublicIndex(), "inp2");

  auto operationId = context.create<AddOperation>();
  auto node = graph.create<Node>(operationId, "add");

  graph.connect(inp1, node, AddOperation::LEFT);
  graph.connect(inp2, node, AddOperation::RIGHT);

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

TEST_F(OperationTest, AddOp2D) {
  core::Context context;

  auto graph = context.create<Graph>("mainGraph");

  std::vector<float> data1{1, 2, 3, 4};
  std::vector<float> data2{7, 8, 9, 10};
  std::vector<float> target{8, 10, 12, 14};

  TensorShape shape{2, 2};
  size_t size = shape.getTotalSize();

  auto loader1 = context.create<loaders::MemcpyLoader>(
      data1.data(), data1.size() * sizeof(float));

  auto loader2 = context.create<loaders::MemcpyLoader>(
      data2.data(), data2.size() * sizeof(float));

  auto inp1 = graph.create<InputNode>(shape, DataType::FLOAT, false,
                                      loader1.getPublicIndex(), "inp1");
  auto inp2 = graph.create<InputNode>(shape, DataType::FLOAT, false,
                                      loader2.getPublicIndex(), "inp2");

  auto operationId = context.create<AddOperation>();
  auto node = graph.create<Node>(operationId, "add");

  graph.connect(inp1, node, AddOperation::LEFT);
  graph.connect(inp2, node, AddOperation::RIGHT);

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
