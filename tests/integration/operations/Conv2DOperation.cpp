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
#include <polarai/operation/Conv2DOperation.hpp>

#include <gtest/gtest.h>

#include <cmath>

using namespace polarai;
using namespace polarai::core;
using namespace polarai::operation;
using namespace polarai::backend::generic;

TEST_F(OperationTest, Conv2D) {
  core::Context context;

  auto graph = context.create<Graph>("mainGraph");

  std::vector<float> input{1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1};
  std::vector<float> kernel{0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1};
  std::vector<float> target{0.9, 0.9, 0.9, 0.9};

  TensorShape shape{4, 4};
  size_t size = shape.getTotalSize();

  auto loader =
      context.create<loaders::MemcpyLoader>(input.data(), size * sizeof(float));

  auto inp = graph.create<InputNode>(shape, DataType::FLOAT, true,
                                     loader.getPublicIndex(), "inp");

  TensorShape kernelShape{3, 3};
  size_t kersize = kernelShape.getTotalSize();

  auto kerLoader = context.create<loaders::MemcpyLoader>(
      kernel.data(), kersize * sizeof(float));

  auto kernelInp =
      graph.create<InputNode>(kernelShape, DataType::FLOAT, false,
                              kerLoader.getPublicIndex(), "kerinp");
  auto operationId = context.create<Conv2DOperation>();
  auto node = graph.create<Node>(operationId, "conv2d");

  graph.connect(inp, node, Conv2DOperation::INPUT);
  graph.connect(kernelInp, node, Conv2DOperation::KERNEL);

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
