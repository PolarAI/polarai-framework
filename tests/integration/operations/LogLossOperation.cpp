#include "OperationTest.hpp"

#include <polarai/core/context/Context.hpp>
#include <polarai/core/node/InputNode.hpp>
#include <polarai/core/node/Node.hpp>
#include <polarai/core/node/OutputNode.hpp>
#include <polarai/loaders/MemcpyLoader.hpp>
#include <polarai/operation/LogLossOperation.hpp>

#include <gtest/gtest.h>

#include <cmath>

using namespace polarai;
using namespace polarai::core;
using namespace polarai::operation;
using namespace polarai::backend::generic;

template <typename T>
std::vector<T> logLossOperation(const std::vector<T>& prediction,
                                const std::vector<T>& groundTruth) {
  constexpr T eps = 1e-5;
  std::vector<T> res(prediction.size());
  for (size_t index = 0; index < prediction.size(); ++index) {
    res[index] =
        -groundTruth[index] * std::log(prediction[index] + eps) -
        (1 - groundTruth[index]) * std::log(1 - prediction[index] + eps);
  }
  return res;
}

TEST_F(OperationTest, DISABLED_LogLoss) {
  core::Context context;

  auto graph = context.create<Graph>("mainGraph");

  std::vector<float> prediction{0.6, 0.2, 0.8, 0.3};
  std::vector<float> groundTruth{1, 0, 1, 0};
  auto target = logLossOperation(prediction, groundTruth);

  TensorShape shape{2, 2};
  size_t size = shape.getTotalSize();

  auto loader1 = context.create<loaders::MemcpyLoader>(prediction.data(),
                                                       size * sizeof(float));

  auto loader2 = context.create<loaders::MemcpyLoader>(groundTruth.data(),
                                                       size * sizeof(float));

  auto inp1 = graph.create<InputNode>(shape, DataType::FLOAT, false,
                                      loader1.getPublicIndex(), "inp1");
  auto inp2 = graph.create<InputNode>(shape, DataType::FLOAT, false,
                                      loader2.getPublicIndex(), "inp2");

  auto operationId = context.create<LogLossOperation>();
  auto node = graph.create<Node>(operationId, "logloss");

  graph.connect(inp1, node, LogLossOperation::PREDICTED);
  graph.connect(inp2, node, LogLossOperation::GROUND_TRUTH);

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
