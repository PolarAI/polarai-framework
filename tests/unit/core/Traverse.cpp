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
#include <gtest/gtest.h>
#include <traversal/ContentChecker.h>
#include <traversal/EdgesChecker.h>
#include <traversal/TopologyChecker.h>

using namespace athena;
using namespace athena::core;
using namespace athena::operation;
using namespace athena::tests::unit;

namespace {
TEST(Traverse, Topology1) {
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
  auto& traversal = graph.traverse();
  auto target =
      std::vector<std::set<utils::Index>>{{inp1, inp2}, {node}, {out}};
  ASSERT_TRUE(checkTraversalContent(traversal, target));
  auto edges = std::set<Edge>{{inp1, node, AddOperation::LEFT},
                              {inp2, node, AddOperation::RIGHT},
                              {node, out, Operation::Unmarked}};
  ASSERT_TRUE(checkEdges(traversal, edges));
  ASSERT_TRUE(checkTopology(traversal));
}

TEST(Traverse, Topology2) {
  Context context;
  auto graph = context.create<Graph>();
  auto inp1 =
      graph.create<InputNode>(TensorShape{2, 2}, DataType::FLOAT, false, 0);
  auto inp2 =
      graph.create<InputNode>(TensorShape{2, 2}, DataType::FLOAT, false, 0);
  auto operationId = context.create<AddOperation>();
  auto node1 = graph.create<Node>(operationId);
  graph.connect(inp1, node1, AddOperation::LEFT);
  graph.connect(inp2, node1, AddOperation::RIGHT);
  auto inp3 =
      graph.create<InputNode>(TensorShape{2, 2}, DataType::FLOAT, false, 0);
  auto node2 = graph.create<Node>(operationId);
  graph.connect(inp3, node2, AddOperation::LEFT);
  graph.connect(node1, node2, AddOperation::RIGHT);
  auto out = graph.create<OutputNode>();
  graph.connect(node2, out, Operation::Unmarked);
  auto& traversal = graph.traverse();
  auto target = std::vector<std::set<utils::Index>>{
      {inp1, inp2, inp3}, {node1}, {node2}, {out}};
  ASSERT_TRUE(checkTraversalContent(traversal, target));
  auto edges = std::set<Edge>{{inp1, node1, AddOperation::LEFT},
                              {inp2, node1, AddOperation::RIGHT},
                              {inp3, node2, AddOperation::LEFT},
                              {node1, node2, AddOperation::RIGHT},
                              {node2, out, Operation::Unmarked}};
  ASSERT_TRUE(checkEdges(traversal, edges));
  ASSERT_TRUE(checkTopology(traversal));
}
} // namespace
