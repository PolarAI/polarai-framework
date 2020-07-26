#include "OperationTest.hpp"

#include <polarai/core/context/Context.hpp>
#include <polarai/core/node/InputNode.hpp>
#include <polarai/core/node/Node.hpp>
#include <polarai/core/node/OutputNode.hpp>
#include <polarai/loaders/MemcpyLoader.hpp>
#include <polarai/operation/MatMulOperation.hpp>

#include <gtest/gtest.h>

using namespace polarai;
using namespace polarai::core;
using namespace polarai::operation;
using namespace polarai::backend::generic;

template <typename T>
std::vector<T>
matMulOperation(const bool leftTranspose, const bool rightTranspose, uint64_t m,
                uint64_t n, uint64_t k, const std::vector<T>& left,
                const std::vector<T>& right) {
  std::vector<T> res(m * n);
  for (uint64_t indexRow = 0; indexRow < m; ++indexRow) {
    for (uint64_t indexColumn = 0; indexColumn < n; ++indexColumn) {
      unsigned long leftIncrement = 0;
      unsigned long leftInd = 0;
      if (leftTranspose == false) { // Not transposed
        leftIncrement = 1;
        leftInd = indexRow * k;
      } else { // Transposed
        leftIncrement = m;
        leftInd = indexRow;
      }
      unsigned long rightIncrement = 0;
      unsigned long rightInd = 0;
      if (rightTranspose == false) { // Not transposed
        rightIncrement = n;
        rightInd = indexColumn;
      } else { // Transposed
        rightIncrement = 1;
        rightInd = indexColumn * k;
      }
      res[indexRow * n + indexColumn] = 0;
      for (int iteration = 0; iteration < k;
           ++iteration, leftInd += leftIncrement, rightInd += rightIncrement) {
        res[indexRow * n + indexColumn] += left[leftInd] * right[rightInd];
      }
    }
  }
  return res;
}

TEST_F(OperationTest, MatMulNN) {
  core::Context context;

  auto graph = context.create<Graph>("mainGraph");

  std::vector<float> left{3, 4, 5, 10};
  std::vector<float> right{6, 7, 8, 11};
  uint64_t m = 1, n = 1, k = 4;
  auto target = matMulOperation(false, false, m, n, k, left, right);

  TensorShape shapeLeft{m, k};
  TensorShape shapeRight{k, n};

  auto loader1 = context.create<loaders::MemcpyLoader>(
      left.data(), left.size() * sizeof(float));

  auto loader2 = context.create<loaders::MemcpyLoader>(
      right.data(), right.size() * sizeof(float));

  auto inp1 = graph.create<InputNode>(shapeLeft, DataType::FLOAT, false,
                                      loader1.getPublicIndex(), "inp1");
  auto inp2 = graph.create<InputNode>(shapeRight, DataType::FLOAT, false,
                                      loader2.getPublicIndex(), "inp2");

  auto operationId = context.create<MatMulOperation>(false, false);
  auto node = graph.create<Node>(operationId, "matmul");

  graph.connect(inp1, node, MatMulOperation::LEFT);
  graph.connect(inp2, node, MatMulOperation::RIGHT);

  auto out = graph.create<OutputNode>("out");
  graph.connect(node, out, Operation::Unmarked);

  withEachDeviceDo(
      [&graph, out, &context, &target, m, n](Executor& executor) {
        executor.addGraph(graph);
        executor.evaluate(graph);

        auto accessor =
            context.internal()->getRef<OutputNode>(out).getAccess<float>(
                executor.getAllocator());
        for (int i = 0; i < m * n; i++) {
          EXPECT_FLOAT_EQ(accessor(i), target[i]);
        }
      });
}

TEST_F(OperationTest, MatMulNNSquare) {
  core::Context context;

  auto graph = context.create<Graph>("mainGraph");

  std::vector<float> left{3, 4, 5, 10};
  std::vector<float> right{6, 7, 8, 11};
  uint64_t m = 2, n = 2, k = 2;
  auto target = matMulOperation(false, false, m, n, k, left, right);

  TensorShape shapeLeft{m, k};
  TensorShape shapeRight{k, n};

  auto loader1 = context.create<loaders::MemcpyLoader>(
      left.data(), left.size() * sizeof(float));

  auto loader2 = context.create<loaders::MemcpyLoader>(
      right.data(), right.size() * sizeof(float));

  auto inp1 = graph.create<InputNode>(shapeLeft, DataType::FLOAT, false,
                                      loader1.getPublicIndex(), "inp1");
  auto inp2 = graph.create<InputNode>(shapeRight, DataType::FLOAT, false,
                                      loader2.getPublicIndex(), "inp2");

  auto operationId = context.create<MatMulOperation>(false, false);
  auto node = graph.create<Node>(operationId, "matmul");

  graph.connect(inp1, node, MatMulOperation::LEFT);
  graph.connect(inp2, node, MatMulOperation::RIGHT);

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

TEST_F(OperationTest, MatMulNNRect) {
  core::Context context;

  auto graph = context.create<Graph>("mainGraph");

  std::vector<float> left{3, 4, 5, 10, 6, 7, 12, 6};
  std::vector<float> right{6, 7, 8, 11, 7, 8, 9, 1, 3, 2, 5, 6};
  uint64_t m = 2, n = 3, k = 4;
  auto target = matMulOperation(false, false, m, n, k, left, right);

  TensorShape shapeLeft{m, k};
  TensorShape shapeRight{k, n};

  auto loader1 = context.create<loaders::MemcpyLoader>(
      left.data(), left.size() * sizeof(float));

  auto loader2 = context.create<loaders::MemcpyLoader>(
      right.data(), right.size() * sizeof(float));

  auto inp1 = graph.create<InputNode>(shapeLeft, DataType::FLOAT, false,
                                      loader1.getPublicIndex(), "inp1");
  auto inp2 = graph.create<InputNode>(shapeRight, DataType::FLOAT, false,
                                      loader2.getPublicIndex(), "inp2");

  auto operationId = context.create<MatMulOperation>(false, false);
  auto node = graph.create<Node>(operationId, "matmul");

  graph.connect(inp1, node, MatMulOperation::LEFT);
  graph.connect(inp2, node, MatMulOperation::RIGHT);

  auto out = graph.create<OutputNode>("out");
  graph.connect(node, out, Operation::Unmarked);

  withEachDeviceDo(
      [&graph, out, &context, &target, m, n](Executor& executor) {
        executor.addGraph(graph);
        executor.evaluate(graph);

        auto accessor =
            context.internal()->getRef<OutputNode>(out).getAccess<float>(
                executor.getAllocator());
        for (int i = 0; i < m * n; i++) {
          EXPECT_FLOAT_EQ(accessor(i), target[i]);
        }
      });
}

TEST_F(OperationTest, MatMulTNRect) {
  core::Context context;

  auto graph = context.create<Graph>("mainGraph");

  std::vector<float> left{3, 6, 4, 7, 5, 12, 10, 6};
  std::vector<float> right{6, 7, 8, 11, 7, 8, 9, 1, 3, 2, 5, 6};
  uint64_t m = 2, n = 3, k = 4;
  auto target = matMulOperation(true, false, m, n, k, left, right);

  TensorShape shapeLeft{k, m};
  TensorShape shapeRight{k, n};

  auto loader1 = context.create<loaders::MemcpyLoader>(
      left.data(), left.size() * sizeof(float));

  auto loader2 = context.create<loaders::MemcpyLoader>(
      right.data(), right.size() * sizeof(float));

  auto inp1 = graph.create<InputNode>(shapeLeft, DataType::FLOAT, false,
                                      loader1.getPublicIndex(), "inp1");
  auto inp2 = graph.create<InputNode>(shapeRight, DataType::FLOAT, false,
                                      loader2.getPublicIndex(), "inp2");

  auto operationId = context.create<MatMulOperation>(true, false);
  auto node = graph.create<Node>(operationId, "matmul");

  graph.connect(inp1, node, MatMulOperation::LEFT);
  graph.connect(inp2, node, MatMulOperation::RIGHT);

  auto out = graph.create<OutputNode>("out");
  graph.connect(node, out, Operation::Unmarked);
  withEachDeviceDo(
      [&graph, out, &context, &target, m, n](Executor& executor) {
        executor.addGraph(graph);
        executor.evaluate(graph);

        auto accessor =
            context.internal()->getRef<OutputNode>(out).getAccess<float>(
                executor.getAllocator());
        for (int i = 0; i < m * n; i++) {
          EXPECT_FLOAT_EQ(accessor(i), target[i]);
        }
      });
}

TEST_F(OperationTest, MatMulNTRect) {
  core::Context context;

  auto graph = context.create<Graph>("mainGraph");

  std::vector<float> left{3, 4, 5, 10, 6, 7, 12, 6};
  std::vector<float> right{6, 11, 9, 2, 7, 7, 1, 5, 8, 8, 3, 6};
  uint64_t m = 2, n = 3, k = 4;
  auto target = matMulOperation(false, true, m, n, k, left, right);

  TensorShape shapeLeft{m, k};
  TensorShape shapeRight{n, k};

  auto loader1 = context.create<loaders::MemcpyLoader>(
      left.data(), left.size() * sizeof(float));

  auto loader2 = context.create<loaders::MemcpyLoader>(
      right.data(), right.size() * sizeof(float));

  auto inp1 = graph.create<InputNode>(shapeLeft, DataType::FLOAT, false,
                                      loader1.getPublicIndex(), "inp1");
  auto inp2 = graph.create<InputNode>(shapeRight, DataType::FLOAT, false,
                                      loader2.getPublicIndex(), "inp2");

  auto operationId = context.create<MatMulOperation>(false, true);
  auto node = graph.create<Node>(operationId, "matmul");

  graph.connect(inp1, node, MatMulOperation::LEFT);
  graph.connect(inp2, node, MatMulOperation::RIGHT);

  auto out = graph.create<OutputNode>("out");
  graph.connect(node, out, Operation::Unmarked);

  withEachDeviceDo(
      [&graph, out, &context, &target, m, n](Executor& executor) {
        executor.addGraph(graph);
        executor.evaluate(graph);

        auto accessor =
            context.internal()->getRef<OutputNode>(out).getAccess<float>(
                executor.getAllocator());
        for (int i = 0; i < m * n; i++) {
          EXPECT_FLOAT_EQ(accessor(i), target[i]);
        }
      });
}
