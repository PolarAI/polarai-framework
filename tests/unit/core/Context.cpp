/*
 * Copyright (c) 2018 Athena. All rights reserved.
 * https://getathena.ml
 *
 * Licensed under MIT license.
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an “AS IS” BASIS, WITHOUT
 * WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the
 * License for the specific language governing permissions and limitations under
 * the License.
 */

#include <athena/core/context/Context.h>
#include <athena/core/graph/Graph.h>
#include <athena/core/node/InputNode.h>
#include <athena/core/node/Node.h>
#include <athena/operation/AddOperation.h>
#include <athena/utils/logger/log.h>

#include <gtest/gtest.h>

using namespace athena::core;

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
  auto operationId = context.create<athena::operation::AddOperation>();
  ASSERT_TRUE(operationId == 5);
  ASSERT_TRUE(context.create<Node>(operationId) == 6);
  ASSERT_TRUE(context.create<InputNode>(TensorShape{3, 2}, DataType::FLOAT,
                                        false, 0) == 7);
}
} // namespace
