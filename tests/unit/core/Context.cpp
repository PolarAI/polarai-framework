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

#include <polarai/core/context/Context.hpp>
#include <polarai/core/graph/Graph.hpp>
#include <polarai/core/node/InputNode.hpp>
#include <polarai/core/node/Node.hpp>
#include <polarai/operation/AddOperation.hpp>
#include <polarai/utils/logger/log.hpp>

#include <gtest/gtest.h>

using namespace polarai::core;

namespace {
TEST(Context, Creation) { Context context; }

TEST(Context, CreationIntoContext) {
  Context context;
  auto graph = context.create<Graph>();
  ASSERT_TRUE(graph.getPublicIndex() == 1);
  graph = context.create<Graph>();
  ASSERT_TRUE(graph.getPublicIndex() == 2);
  graph = context.create<Graph>();
  ASSERT_TRUE(graph.getPublicIndex() == 3);
  graph = context.create<Graph>();
  ASSERT_TRUE(graph.getPublicIndex() == 4);
  auto operationId = context.create<polarai::operation::AddOperation>();
  ASSERT_TRUE(operationId == 5);
  ASSERT_TRUE(context.create<Node>(operationId) == 6);
  ASSERT_TRUE(context.create<InputNode>(TensorShape{3, 2}, DataType::FLOAT,
                                        false, 0) == 7);
}
} // namespace
