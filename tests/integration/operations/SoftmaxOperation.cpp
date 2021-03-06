//===----------------------------------------------------------------------===//
// Copyright (c) 2020 PolarAI. All rights reserved.
//
// Licensed under MIT license.
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS, WITHOUT
// WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the
// License for the specific language governing permissions and limitations under
// the License.
//===----------------------------------------------------------------------===//

#include "OperationTest.hpp"

#include <polarai/core/context/Context.hpp>
#include <polarai/core/node/InputNode.hpp>
#include <polarai/core/node/Node.hpp>
#include <polarai/core/node/OutputNode.hpp>
#include <polarai/loaders/MemcpyLoader.hpp>
#include <polarai/operation/SoftmaxOperation.hpp>

#include <gtest/gtest.h>

#include <cmath>

using namespace polarai;
using namespace polarai::core;
using namespace polarai::operation;
using namespace polarai::backend::generic;

TEST_F(OperationTest, Softmax) {
  core::Context context;

  auto graph = context.create<Graph>("mainGraph");

  std::vector<float> input{1, 2, 3, 4};
  std::vector<float> target{0.0320586, 0.08714432, 0.23688282, 0.64391426};

  TensorShape shape{2, 2};
  size_t size = shape.getTotalSize();

  auto loader =
      context.create<loaders::MemcpyLoader>(input.data(), size * sizeof(float));

  auto inp = graph.create<InputNode>(shape, DataType::FLOAT, false,
                                     loader.getPublicIndex(), "inp");

  auto operationId = context.create<SoftmaxOperation>();
  auto node = graph.create<Node>(operationId, "softmax");

  graph.connect(inp, node, SoftmaxOperation::Unmarked);

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
