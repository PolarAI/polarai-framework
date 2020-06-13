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

#include <athena/backend/llvm/LLVMExecutor.h>
#include <athena/backend/llvm/runtime/GraphHandle.h>
#include <athena/core/context/Context.h>
#include <athena/core/graph/Graph.h>
#include <athena/core/node/InputNode.h>
#include <athena/core/node/Node.h>
#include <athena/core/node/OutputNode.h>
#include <athena/loaders/MemcpyLoader.h>
#include <athena/operation/AddOperation.h>
#include <athena/operation/CombineOperation.h>
#include <athena/operation/DivideOperation.h>
#include <athena/operation/LogLossOperation.h>
#include <athena/operation/MulOperation.h>
#include <athena/operation/SigmoidOperation.h>
#include <athena/operation/MatMulOperation.h>
#include <athena/operation/MulConcatOperation.h>

#include <gtest/gtest.h>

#include <cmath>
#include <vector>
#include <cmath>


#include <fstream>

#include <unistd.h>

using namespace athena;
using namespace athena::core;
using namespace athena::operation;
using namespace athena::backend::llvm;

namespace {
const float eps = 1e-5;

TEST(JITIntegration, DISABLED_AddOperationSample) {
  Context context;

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

  LLVMExecutor executor;
  executor.addGraph(graph);
  executor.evaluate(graph);

  auto& allocator = executor.getAllocator();
  MemoryRecord record;
  auto tensorPtr =
      context.internal()->getRef<AbstractNodeInternal>(node).getTensorPtr();
  record.virtualAddress = tensorPtr->getVirtualAddress();
  record.allocationSize = size * sizeof(float);
  allocator.lock(record, LockType::READ);
  auto* res = static_cast<float*>(allocator.get(record));
  for (int i = 0; i < size; i++) {
    EXPECT_NEAR(res[i], target[i], eps);
  }
  allocator.release(record);

  std::vector<float> data3{6, 7, 8, 9};
  std::vector<float> data4{3, 4, 5, 6};
  std::vector<float> target2{9, 11, 13, 15};
  loader1.setPointer(data3.data(), data3.size() * sizeof(float));
  loader2.setPointer(data4.data(), data4.size() * sizeof(float));
  executor.evaluate(graph);

  allocator.lock(record, LockType::READ);
  res = static_cast<float*>(allocator.get(record));
  for (int i = 0; i < size; i++) {
    EXPECT_NEAR(res[i], target2[i], eps);
  }
  allocator.release(record);
  std::cout << "**************************************************"
            << std::endl;
}

template <typename T>
std::vector<T> combineOperation(T alpha, const std::vector<T>& left, T beta,
                                const std::vector<T>& right) {
  std::vector<T> res(left.size());
  for (size_t index = 0; index < left.size(); ++index) {
    res[index] = alpha * left[index] + beta * right[index];
  }
  return res;
}

TEST(JITIntegration, DISABLED_CombineOperationSample) {
  Context context;

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

  LLVMExecutor executor;
  executor.addGraph(graph);
  executor.evaluate(graph);

  auto& allocator = executor.getAllocator();
  MemoryRecord record;
  auto tensorPtr =
      context.internal()->getRef<AbstractNodeInternal>(node).getTensorPtr();
  record.virtualAddress = tensorPtr->getVirtualAddress();
  record.allocationSize = size * sizeof(float);
  allocator.lock(record, LockType::READ);
  auto* res = static_cast<float*>(allocator.get(record));
  for (int i = 0; i < size; i++) {
    EXPECT_FLOAT_EQ(res[i], target[i]);
  }
  allocator.release(record);

  std::vector<float> data3{6, 7, 8, 9};
  std::vector<float> data4{3, 4, 5, 6};
  auto target2 = combineOperation(alpha, data3, beta, data4);
  loader1.setPointer(data3.data(), data3.size() * sizeof(float));
  loader2.setPointer(data4.data(), data4.size() * sizeof(float));
  executor.evaluate(graph);

  allocator.lock(record, LockType::READ);
  res = static_cast<float*>(allocator.get(record));
  for (int i = 0; i < size; i++) {
    EXPECT_NEAR(res[i], target2[i], eps);
  }
  allocator.release(record);
}

template <typename T>
std::vector<T> divideOperation(const std::vector<T>& numerator,
                               const std::vector<T>& denominator) {
  std::vector<T> res(numerator.size());
  for (size_t index = 0; index < numerator.size(); ++index) {
    res[index] = numerator[index] / denominator[index];
  }
  return res;
}

TEST(JITIntegration, DISABLED_DivideOperationSample) {
  Context context;

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

  LLVMExecutor executor;
  executor.addGraph(graph);
  executor.evaluate(graph);

  auto& allocator = executor.getAllocator();
  MemoryRecord record;
  auto tensorPtr =
      context.internal()->getRef<AbstractNodeInternal>(node).getTensorPtr();
  record.virtualAddress = tensorPtr->getVirtualAddress();
  record.allocationSize = size * sizeof(float);
  allocator.lock(record, LockType::READ);
  auto* res = static_cast<float*>(allocator.get(record));
  for (int i = 0; i < size; i++) {
    EXPECT_NEAR(res[i], target[i], eps);
  }
  allocator.release(record);

  std::vector<float> data3{6, 7, 8, 9};
  std::vector<float> data4{3, 4, 5, 6};
  auto target2 = divideOperation(data3, data4);
  loader1.setPointer(data3.data(), data3.size() * sizeof(float));
  loader2.setPointer(data4.data(), data4.size() * sizeof(float));
  executor.evaluate(graph);

  allocator.lock(record, LockType::READ);
  res = static_cast<float*>(allocator.get(record));
  for (int i = 0; i < size; i++) {
    EXPECT_NEAR(res[i], target2[i], eps);
  }
  allocator.release(record);
}

template <typename T>
std::vector<T> logLossOperation(const std::vector<T>& prediction,
                                const std::vector<T>& groundTruth) {
  std::vector<T> res(prediction.size());
  for (size_t index = 0; index < prediction.size(); ++index) {
    res[index] =
        -groundTruth[index] * std::log(prediction[index] + eps) -
        (1 - groundTruth[index]) * std::log(1 - prediction[index] + eps);
  }
  return res;
}

TEST(JITIntegration, DISABLED_LogLossOperationSample) {
  Context context;

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

  LLVMExecutor executor;
  executor.addGraph(graph);
  executor.evaluate(graph);

  auto& allocator = executor.getAllocator();
  MemoryRecord record;
  auto tensorPtr =
      context.internal()->getRef<AbstractNodeInternal>(node).getTensorPtr();
  record.virtualAddress = tensorPtr->getVirtualAddress();
  record.allocationSize = size * sizeof(float);
  allocator.lock(record, LockType::READ);
  auto* res = static_cast<float*>(allocator.get(record));
  for (int i = 0; i < size; i++) {
    EXPECT_NEAR(res[i], target[i], eps);
  }
  allocator.release(record);

  std::vector<float> prediction2{0.7, 0.3, 0.2, 0.9};
  std::vector<float> groundTruth2{1, 0, 0, 1};
  auto target2 = logLossOperation(prediction2, groundTruth2);

  loader1.setPointer(prediction2.data(), prediction2.size() * sizeof(float));
  loader2.setPointer(groundTruth2.data(), groundTruth2.size() * sizeof(float));
  executor.evaluate(graph);

  allocator.lock(record, LockType::READ);
  res = static_cast<float*>(allocator.get(record));
  for (int i = 0; i < size; i++) {
    EXPECT_NEAR(res[i], target2[i], eps);
  }
  allocator.release(record);
}

template <typename T>
std::vector<T> mulOperation(const std::vector<T>& left,
                            const std::vector<T>& right) {
  std::vector<T> res(left.size());
  for (size_t index = 0; index < left.size(); ++index) {
    res[index] = left[index] * right[index];
  }
  return res;
}

TEST(JITIntegration, DISABLED_MulOperationSample) {
  Context context;

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

  LLVMExecutor executor;
  executor.addGraph(graph);
  executor.evaluate(graph);

  auto& allocator = executor.getAllocator();
  MemoryRecord record;
  auto tensorPtr =
      context.internal()->getRef<AbstractNodeInternal>(node).getTensorPtr();
  record.virtualAddress = tensorPtr->getVirtualAddress();
  record.allocationSize = size * sizeof(float);
  allocator.lock(record, LockType::READ);
  auto* res = static_cast<float*>(allocator.get(record));
  for (int i = 0; i < size; i++) {
    EXPECT_NEAR(res[i], target[i], eps);
  }
  allocator.release(record);

  std::vector<float> left2{120.7, 410.3, 410.2, 220.9};
  std::vector<float> right2{17, 0, 31.20, 21};
  auto target2 = mulOperation(left2, right2);

  loader1.setPointer(left2.data(), left2.size() * sizeof(float));
  loader2.setPointer(right2.data(), right2.size() * sizeof(float));
  executor.evaluate(graph);

  allocator.lock(record, LockType::READ);
  res = static_cast<float*>(allocator.get(record));
  for (int i = 0; i < size; i++) {
    EXPECT_NEAR(res[i], target2[i], eps);
  }
  allocator.release(record);
}

template <typename T>
std::vector<T>
matMulOperation(const bool leftTranspose, const bool rightTranspose, uint64_t m,
                uint64_t n, uint64_t k, const std::vector<T>& left,
                const std::vector<T>& right) {
  std::vector<T> res(m * n);
  for (uint64_t indexRow = 0; indexRow < m; ++indexRow) {
    for (uint64_t indexColumn = 0; indexColumn < n; ++indexColumn) {
      ulong leftIncrement = 0;
      ulong leftInd = 0;
      if (leftTranspose == false) { // Not transposed
        leftIncrement = 1;
        leftInd = indexRow * k;
      } else { // Transposed
        leftIncrement = m;
        leftInd = indexRow;
      }
      ulong rightIncrement = 0;
      ulong rightInd = 0;
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

TEST(JITIntegration, DISABLED_MatMulNNOperationSample) {
  Context context;

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

  LLVMExecutor executor;
  executor.addGraph(graph);
  executor.evaluate(graph);

  auto& allocator = executor.getAllocator();
  uint64_t res_size = m * n;
  MemoryRecord record;
  auto tensorPtr =
      context.internal()->getRef<AbstractNodeInternal>(node).getTensorPtr();
  record.virtualAddress = tensorPtr->getVirtualAddress();
  record.allocationSize = res_size * sizeof(float);
  allocator.lock(record, LockType::READ);
  auto* res = static_cast<float*>(allocator.get(record));
  for (int i = 0; i < res_size; i++) {
    EXPECT_NEAR(res[i], target[i], eps);
  }
  allocator.release(record);

  std::vector<float> left2{1, 2, 3, 4};
  std::vector<float> right2{16, 17, 18, 111};
  auto target2 = matMulOperation(false, false, m, n, k, left2, right2);

  loader1.setPointer(left2.data(), left2.size() * sizeof(float));
  loader2.setPointer(right2.data(), right2.size() * sizeof(float));
  executor.evaluate(graph);

  allocator.lock(record, LockType::READ);
  res = static_cast<float*>(allocator.get(record));
  for (int i = 0; i < res_size; i++) {
    EXPECT_NEAR(res[i], target2[i], eps);
  }
  allocator.release(record);
}

TEST(JITIntegration, DISABLED_MatMulNNSquareOperationSample) {
  Context context;

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

  LLVMExecutor executor;
  executor.addGraph(graph);
  executor.evaluate(graph);

  auto& allocator = executor.getAllocator();
  uint64_t res_size = m * n;
  MemoryRecord record;
  auto tensorPtr =
      context.internal()->getRef<AbstractNodeInternal>(node).getTensorPtr();
  record.virtualAddress = tensorPtr->getVirtualAddress();
  record.allocationSize = res_size * sizeof(float);
  allocator.lock(record, LockType::READ);
  auto* res = static_cast<float*>(allocator.get(record));
  for (int i = 0; i < res_size; i++) {
    EXPECT_NEAR(res[i], target[i], eps);
  }
  allocator.release(record);

  std::vector<float> left2{1, 2, 3, 4};
  std::vector<float> right2{16, 17, 18, 111};
  auto target2 = matMulOperation(false, false, m, n, k, left2, right2);

  loader1.setPointer(left2.data(), left2.size() * sizeof(float));
  loader2.setPointer(right2.data(), right2.size() * sizeof(float));
  executor.evaluate(graph);

  allocator.lock(record, LockType::READ);
  res = static_cast<float*>(allocator.get(record));
  for (int i = 0; i < res_size; i++) {
    EXPECT_NEAR(res[i], target2[i], eps);
  }
  allocator.release(record);
}

TEST(JITIntegration, DISABLED_MatMulNNRectOperationSample) {
  Context context;

  auto graph = context.create<Graph>("mainGraph");

  std::vector<float> left{
      3, 4, 5, 10,
      6, 7, 12, 6};
  std::vector<float> right{
      6, 7, 8,
      11, 7, 8,
      9, 1, 3,
      2, 5, 6
  };
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

  LLVMExecutor executor;
  executor.addGraph(graph);
  executor.evaluate(graph);

  auto& allocator = executor.getAllocator();
  uint64_t res_size = m * n;
  MemoryRecord record;
  auto tensorPtr =
      context.internal()->getRef<AbstractNodeInternal>(node).getTensorPtr();
  record.virtualAddress = tensorPtr->getVirtualAddress();
  record.allocationSize = res_size * sizeof(float);
  allocator.lock(record, LockType::READ);
  auto* res = static_cast<float*>(allocator.get(record));
  for (int i = 0; i < res_size; i++) {
    EXPECT_NEAR(res[i], target[i], eps);
  }
  // todo check shape
  allocator.release(record);
  std::vector<float> left2{
      31, 24, 0, 10,
      63, 47, -12, 6
  };
  std::vector<float> right2{
      -166, 27, 8,
      11, 57, 8,
      9, -1, 3,
      32, 5, 26
  };
  auto target2 = matMulOperation(false, false, m, n, k, left2, right2);

  loader1.setPointer(left2.data(), left2.size() * sizeof(float));
  loader2.setPointer(right2.data(), right2.size() * sizeof(float));
  executor.evaluate(graph);

  allocator.lock(record, LockType::READ);
  res = static_cast<float*>(allocator.get(record));
  for (int i = 0; i < res_size; i++) {
    EXPECT_NEAR(res[i], target2[i], eps);
  }
  // todo check shape
  auto tmpShape = context.internal()
                      ->getRef<AbstractNodeInternal>(node)
                      .getTensorPtr()
                      ->getShape();
  allocator.release(record);
}

TEST(JITIntegration, DISABLED_MatMulTNRectOperationSample) {
  Context context;

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

  LLVMExecutor executor;
  executor.addGraph(graph);
  executor.evaluate(graph);

  auto& allocator = executor.getAllocator();
  uint64_t res_size = m * n;
  MemoryRecord record;
  auto tensorPtr =
      context.internal()->getRef<AbstractNodeInternal>(node).getTensorPtr();
  record.virtualAddress = tensorPtr->getVirtualAddress();
  record.allocationSize = res_size * sizeof(float);
  allocator.lock(record, LockType::READ);
  auto* res = static_cast<float*>(allocator.get(record));
  for (int i = 0; i < res_size; i++) {
    EXPECT_NEAR(res[i], target[i], eps);
  }
  // todo check shape
  auto tmpShape = context.internal()
                      ->getRef<AbstractNodeInternal>(node)
                      .getTensorPtr()
                      ->getShape();
  allocator.release(record);
}

TEST(JITIntegration, DISABLED_MatMulNTRectOperationSample) {
  Context context;

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

  LLVMExecutor executor;
  executor.addGraph(graph);
  executor.evaluate(graph);

  auto& allocator = executor.getAllocator();
  uint64_t res_size = m * n;
  MemoryRecord record;
  auto tensorPtr =
      context.internal()->getRef<AbstractNodeInternal>(node).getTensorPtr();
  record.virtualAddress = tensorPtr->getVirtualAddress();
  record.allocationSize = res_size * sizeof(float);
  allocator.lock(record, LockType::READ);
  auto* res = static_cast<float*>(allocator.get(record));
  for (int i = 0; i < res_size; i++) {
    EXPECT_NEAR(res[i], target[i], eps);
  }
  // todo check shape
  auto tmpShape = context.internal()
                      ->getRef<AbstractNodeInternal>(node)
                      .getTensorPtr()
                      ->getShape();
  allocator.release(record);
}

template <typename T>
std::vector<T> sigmoidOperation(const std::vector<T>& input) {
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

TEST(JITIntegration, DISABLED_SigmoidOperationSample) {
  Context context;

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

  LLVMExecutor executor;
  executor.addGraph(graph);
  executor.evaluate(graph);

  auto& allocator = executor.getAllocator();
  MemoryRecord record;
  auto tensorPtr =
      context.internal()->getRef<AbstractNodeInternal>(node).getTensorPtr();
  record.virtualAddress = tensorPtr->getVirtualAddress();
  record.allocationSize = size * sizeof(float);
  allocator.lock(record, LockType::READ);
  auto* res = static_cast<float*>(allocator.get(record));
  for (int i = 0; i < size; i++) {
    EXPECT_NEAR(res[i], target[i], eps);
  }
  allocator.release(record);

  std::vector<float> input2{-17, 2, 9, 43};
  auto target2 = sigmoidOperation(input2);

  loader.setPointer(input2.data(), input2.size() * sizeof(float));
  executor.evaluate(graph);

  allocator.lock(record, LockType::READ);
  res = static_cast<float*>(allocator.get(record));
  for (int i = 0; i < size; i++) {
    EXPECT_NEAR(res[i], target2[i], eps);
  }
  allocator.release(record);
}

template <typename T>
std::vector<T> mulConcatOperation(const std::vector<T>& localDerivative, const std::vector<T>& gradient) {
  std::vector<T> res(localDerivative.size());
  for (size_t index = 0; index < localDerivative.size(); ++index) {
    res[index] = localDerivative[index] * gradient[0];
  }
  return res;
}

TEST(JITIntegration, DISABLED_MulConcatOperationSample) {
  Context context;

  auto graph = context.create<Graph>("mainGraph");

  std::vector<float> localDerivative{-100, 2, 5, 50};
  std::vector<float> gradient{17.5};
  auto target = mulConcatOperation(localDerivative, gradient);

  TensorShape shape{2, 2};
  size_t size = shape.getTotalSize();

  auto loaderLocalDerivative =
      context.create<loaders::MemcpyLoader>(localDerivative.data(), size * sizeof(float));
  auto inpLocalDerivative = graph.create<InputNode>(shape, DataType::FLOAT, false,
                                                    loaderLocalDerivative.getPublicIndex(), "inpLocDeriv");

  auto loaderGradient =
      context.create<loaders::MemcpyLoader>(gradient.data(), 1 * sizeof(float));
  auto inpGradient= graph.create<InputNode>(TensorShape{1, 1}, DataType::FLOAT, false,
                                            loaderGradient.getPublicIndex(), "inpGradient");

  auto operationId = context.create<MulConcatOperation>();
  auto node = graph.create<Node>(operationId, "mulconcat");

  graph.connect(inpLocalDerivative, node, MulConcatOperation::LOCAL_DERIVATIVE);
  graph.connect(inpGradient, node, MulConcatOperation::GRADIENT);

  auto out = graph.create<OutputNode>("out");
  graph.connect(node, out, Operation::Unmarked);

  LLVMExecutor executor;
  executor.addGraph(graph);
  executor.evaluate(graph);

  auto& allocator = executor.getAllocator();
  MemoryRecord record;
  auto tensorPtr =
      context.internal()->getRef<AbstractNodeInternal>(node).getTensorPtr();
  record.virtualAddress = tensorPtr->getVirtualAddress();
  record.allocationSize = size * sizeof(float);
  allocator.lock(record, LockType::READ);
  auto* res = static_cast<float*>(allocator.get(record));
  for (int i = 0; i < size; i++) {
    EXPECT_NEAR(res[i], target[i], eps);
  }
  allocator.release(record);

  std::vector<float> localDerivative2{-100, 2, 5, 50};
  std::vector<float> gradient2{17.5};
  auto target2 = mulConcatOperation(localDerivative, gradient);

  loaderLocalDerivative.setPointer(localDerivative2.data(), localDerivative2.size() * sizeof(float));
  loaderGradient.setPointer(gradient2.data(), gradient2.size() * sizeof(float));
  executor.evaluate(graph);

  allocator.lock(record, LockType::READ);
  res = static_cast<float*>(allocator.get(record));
  for (int i = 0; i < size; i++) {
    EXPECT_NEAR(res[i], target2[i], eps);
  }
  allocator.release(record);
}

std::vector<float> readDataset(const char* filename) {
  std::vector<float> res;
  std::string cell;
  std::ifstream ifs(filename);
  while(std::getline(ifs, cell, ','))
  {
    res.emplace_back(atof(cell.c_str()));
  }
  return res;
}

float testModel(LLVMExecutor& executor, Graph& graph, TensorInternal* lossTensorPtr, loaders::MemcpyLoader& inputLoader, loaders::MemcpyLoader& groundTruthLoader,
                std::vector<float>& dataset, size_t sampleSize, size_t rowsCount) {
  float sumError = 0;
  for (size_t index = 0; index < rowsCount; ++index) {
    if (index % 2 == 0) {     // if index % 2 == 0 then sample is in the test dataset, else in the train
      inputLoader.setPointer(dataset.data() + index * (sampleSize + 1),
                             sampleSize * sizeof(float));
      groundTruthLoader.setPointer(
          dataset.data() + index * (sampleSize + 1) + sampleSize, 1 * sizeof(float));
      executor.evaluate(graph);
      MemoryRecord recordLoss;
      auto& allocator = executor.getAllocator();
      recordLoss.virtualAddress = lossTensorPtr->getVirtualAddress();
      recordLoss.allocationSize = 1 * sizeof(float);
      allocator.lock(recordLoss, LockType::READ);
      auto res = static_cast<float*>(allocator.get(recordLoss));
      sumError += res[0];
      allocator.release(recordLoss);
    }
  }
  std::cout << std::endl;
  return sumError;
}


TEST(JITIntegration, DISABLED_TopologyLogReg) {
  Context context;
  auto graph = context.create<Graph>("graph1");
  size_t sampleSize = 3;
  auto dataset = readDataset("dataset_intersect.csv");
  size_t rowsCount = dataset.size() / (sampleSize + 1);

  // Node for data loading
  auto inputLoader =
      context.create<loaders::MemcpyLoader>(nullptr, 0);
  auto dataInputNode = graph.create<InputNode>(
      TensorShape{1, sampleSize}, DataType::FLOAT, true, inputLoader.getPublicIndex(), "inpVector");

  // Node for weights holding
  std::vector<float> weights{0.1, 1, 0.2};
  auto weightsLoader =
      context.create<loaders::MemcpyLoader>(weights.data(), sampleSize * sizeof(float));
  auto weightsInputNode = graph.create<InputNode>(
      TensorShape{sampleSize, 1}, DataType::FLOAT, false, weightsLoader.getPublicIndex(), "weightsVector");

  // MatMulNode
  auto operationMatMulId =
      context.create<MatMulOperation>(false, false, "gemm");
  auto nodeMatMul = graph.create<Node>(operationMatMulId, "nodeGemm");
  graph.connect(dataInputNode, nodeMatMul, MatMulOperation::LEFT);
  graph.connect(weightsInputNode, nodeMatMul, MatMulOperation::RIGHT);

  // Sigmoid node
  auto operationSigmoidId = context.create<SigmoidOperation>("sigmoid");
  auto nodeSigmoid = graph.create<Node>(operationSigmoidId, "nodeSigmoid");
  graph.connect(nodeMatMul, nodeSigmoid, SigmoidOperation::Unmarked);

  // Node for ground truth loading
  std::vector<float> groundTruth{0};
  auto groundTruthLoader =
      context.create<loaders::MemcpyLoader>(groundTruth.data(), groundTruth.size() * sizeof(float));
  auto inpGroundTruth = graph.create<InputNode>(
      TensorShape{1, 1}, DataType::FLOAT, true, groundTruthLoader.getPublicIndex(), "groundTruth");

  // Log loss node
  auto operationLogLossId = context.create<LogLossOperation>("logloss");
  auto loss = graph.create<Node>(operationLogLossId, "loss");

  graph.connect(nodeSigmoid, loss, LogLossOperation::PREDICTED);
  graph.connect(inpGroundTruth, loss, LogLossOperation::GROUND_TRUTH);

  auto [graphGradient, graphConnector] = graph.getGradient(loss);

  LLVMExecutor executor;
  executor.addGraph(graph);
  executor.addGraph(graphGradient);
  executor.addGraph(graphConnector);

  auto& allocator = executor.getAllocator();
  std::ofstream logfile("logfile_intersect.csv");
  for (size_t index = 0; index < 75; ++index) {
    if (index % 2 == 1) {
      auto lossTensorPtr = context.internal()->getRef<AbstractNodeInternal>(loss).getTensorPtr();
      float sumError = testModel(executor, graph, lossTensorPtr, inputLoader, groundTruthLoader, dataset, sampleSize, rowsCount);
      logfile << sumError << ',';
            std::cout << "SumError: " << sumError << std::endl;
      for (int i = 0; i < sampleSize; i++) {
        std::cout << "Weights: ";
        std::cout << weights[i] << ' ';
        logfile << weights[i] << ',';
      }
      logfile << '\n';
      std::cout << std::endl;
      inputLoader.setPointer(dataset.data() + index * (sampleSize + 1),
                             sampleSize * sizeof(float));
      groundTruthLoader.setPointer(
          dataset.data() + index * (sampleSize + 1) + sampleSize, 1 * sizeof(float));
      executor.evaluate(graph);

      std::cout << "**************************************" << std::endl;

      // Optimize
      executor.evaluate(graphGradient);
      executor.evaluate(graphConnector);

      MemoryRecord recordWeights;
      lossTensorPtr = context.internal()
          ->getRef<AbstractNodeInternal>(weightsInputNode)
          .getTensorPtr();
      recordWeights.virtualAddress = lossTensorPtr->getVirtualAddress();
      recordWeights.allocationSize = sampleSize * sizeof(float);
      allocator.lock(recordWeights, LockType::READ);
      auto res = static_cast<float*>(allocator.get(recordWeights));
      for (int i = 0; i < sampleSize; i++) {
        weights[i] = res[i];
      }
      allocator.release(recordWeights);
    }
  }
}
}
