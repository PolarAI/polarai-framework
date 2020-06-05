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

#include <gtest/gtest.h>

#include <vector>

using namespace athena;
using namespace athena::core;
using namespace athena::operation;
using namespace athena::backend::llvm;

TEST(JITIntegration, DISABLED_AddOperationSample) {
  Context context;

  auto graph = context.create<Graph>("mainGraph");

  std::vector<float> data(32, 1);

  auto loader = context.create<loaders::MemcpyLoader>(
      data.data(), data.size() * sizeof(float));

  auto inp1 = graph.create<InputNode>(TensorShape{32}, DataType::FLOAT, false,
                                      loader, "inp1");
  auto inp2 = graph.create<InputNode>(TensorShape{32}, DataType::FLOAT, false,
                                      loader, "inp2");

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
  record.virtualAddress = 257;
  record.allocationSize = 32 * 4;
  allocator.lock(record, LockType::READ);
  auto* res = static_cast<float*>(allocator.get(record));
  for (int i = 0; i < 32; i++) {
    EXPECT_FLOAT_EQ(res[i], 2.0f);
  }
  allocator.release(record);
}
