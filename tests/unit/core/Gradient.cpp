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
#include <athena/core/node/OutputNode.h>
#include <athena/operation/AddOperation.h>
#include <traversal/ContentChecker.h>
#include <traversal/EdgesChecker.h>
#include <traversal/TopologyChecker.h>

#include <gtest/gtest.h>

using namespace athena;
using namespace athena::core;
using namespace athena::operation;
using namespace athena::tests::unit;

namespace {
TEST(Gradient, Topology1) {
  Context context;
  auto graph = context.create<Graph>();
  auto inp1 =
      graph.create<InputNode>(TensorShape{2, 2}, DataType::FLOAT, false, 0);
  auto inp2 =
      graph.create<InputNode>(TensorShape{2, 2}, DataType::FLOAT, false, 0);
  auto operationId = context.create<AddOperation>();
  auto node = graph.create<Node>(operationId);
  graph.connect(inp1, node, AddOperation::LEFT);
  graph.connect(inp2, node, AddOperation::RIGHT);
  auto out = graph.create<OutputNode>();
  graph.connect(node, out, Operation::Unmarked);
  // auto& traversal = graph.traverse();
  // auto [gradientGraph, connectGraph] =
  graph.getGradient();
}
} // namespace
