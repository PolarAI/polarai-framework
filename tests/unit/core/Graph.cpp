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

#include <athena/core/GradientDescent.h>
#include <athena/core/Graph.h>

#include <gtest/gtest.h>
#include <set>

namespace athena::core {

class GraphTest : public ::testing::Test {
protected:
  Context context;
  Graph graph;
  DummyLoader loader;
  OperationDummy operation;

  GraphTest()
      : context(Context()), graph(context), operation("DummyOperation"){};

  void SetUp() override {
    graph.clear();
    graph.setUpOptimizer<Optimizer>();
  }
};

TEST(GraphSimpleTest, Creation) {
  Context context;
  Graph graph(context);
}
TEST_F(GraphTest, Using) {
  InputNode inputNodeFirst({1}, DataType::HALF, loader, context, false,
                           "Input1");
  ASSERT_EQ(inputNodeFirst.getNodeIndex(), 1);
  graph.addNode(inputNodeFirst);
  InputNode inputNodeSecond({1}, DataType::HALF, loader, context, false,
                            "Input2");
  ASSERT_EQ(inputNodeSecond.getNodeIndex(), 2);
  graph.addNode(inputNodeSecond);
  Node node(operation, context, "Node1");
  ASSERT_EQ(node.getNodeIndex(), 3);
  graph.addNode(node);
  inputNodeFirst.before(node, 1);
  inputNodeSecond.before(node, 1);
  Node nodeSecondLayerFirst(operation, context, "Node2.1");
  ASSERT_EQ(nodeSecondLayerFirst.getNodeIndex(), 4);
  Node nodeSecondLayerSecond(operation, context, "Node2.2");
  ASSERT_EQ(nodeSecondLayerSecond.getNodeIndex(), 5);
  graph.addNode(nodeSecondLayerFirst);
  graph.addNode(nodeSecondLayerSecond);
  nodeSecondLayerFirst.after(node, 1);
  nodeSecondLayerSecond.after(node, 1);
  ASSERT_EQ(graph.countOwningNodes(), 0);
  ASSERT_EQ(graph.countSyncNodes(), 5);
  std::vector<inner::Edge> required_topology{
      {inputNodeFirst.getNodeIndex(), node.getNodeIndex(), 1},
      {inputNodeSecond.getNodeIndex(), node.getNodeIndex(), 1},
      {node.getNodeIndex(), nodeSecondLayerFirst.getNodeIndex(), 1},
      {node.getNodeIndex(), nodeSecondLayerSecond.getNodeIndex(), 1}};
  auto topology = graph.getTopology();
  std::sort(required_topology.begin(), required_topology.end());
  std::sort(topology.begin(), topology.end());
  ASSERT_EQ(required_topology, topology);
  node.saveInGraph();
  ASSERT_EQ(graph.countOwningNodes(), 1);
  ASSERT_EQ(graph.countSyncNodes(), 4);
  ASSERT_EQ(required_topology, graph.getTopology());
}
TEST_F(GraphTest, RemoveNode) {
  InputNode inputNodeFirst({1}, DataType::FLOAT, loader, context, false,
                           "Input1");
  ASSERT_EQ(inputNodeFirst.getNodeIndex(), 1);
  graph.addNode(inputNodeFirst);
  InputNode inputNodeSecond({1}, DataType::FLOAT, loader, context, false,
                            "Input2");
  ASSERT_EQ(inputNodeSecond.getNodeIndex(), 2);
  graph.addNode(inputNodeSecond);
  Node node(operation, context, "Node1");
  ASSERT_EQ(node.getNodeIndex(), 3);
  graph.addNode(node);
  inputNodeFirst.before(node, 1);
  inputNodeSecond.before(node, 1);
  Node nodeSecondLayerFirst(operation, context, "Node2.1");
  ASSERT_EQ(nodeSecondLayerFirst.getNodeIndex(), 4);
  Node nodeSecondLayerSecond(operation, context, "Node2.2");
  ASSERT_EQ(nodeSecondLayerSecond.getNodeIndex(), 5);
  graph.addNode(nodeSecondLayerFirst);
  graph.addNode(nodeSecondLayerSecond);
  nodeSecondLayerFirst.after(node, 1);
  nodeSecondLayerSecond.after(node, 1);
  ASSERT_EQ(graph.countOwningNodes(), 0);
  ASSERT_EQ(graph.countSyncNodes(), 5);
  ASSERT_EQ(graph.getTopology().size(), 4);
  node.removeFromGraph();
  ASSERT_EQ(graph.getTopology().size(), 0);
}
TEST_F(GraphTest, DeepTestTraverse) {
  InputNode inputNodeFirst({1}, DataType::DOUBLE, loader, context, false,
                           "Input1");
  ASSERT_EQ(inputNodeFirst.getNodeIndex(), 1);
  graph.addNode(inputNodeFirst);
  InputNode inputNodeSecond({1}, DataType::DOUBLE, loader, context, false,
                            "Input2");
  ASSERT_EQ(inputNodeSecond.getNodeIndex(), 2);
  graph.addNode(inputNodeSecond);
  Node node(operation, context, "Node1");
  ASSERT_EQ(node.getNodeIndex(), 3);
  graph.addNode(node);
  inputNodeFirst.before(node, 1);
  inputNodeSecond.before(node, 1);
  Node nodeSecondLayerFirst(operation, context, "Node2.1");
  ASSERT_EQ(nodeSecondLayerFirst.getNodeIndex(), 4);
  Node nodeSecondLayerSecond(operation, context, "Node2.2");
  ASSERT_EQ(nodeSecondLayerSecond.getNodeIndex(), 5);
  graph.addNode(nodeSecondLayerFirst);
  graph.addNode(nodeSecondLayerSecond);
  nodeSecondLayerFirst.after(node, 1);
  nodeSecondLayerSecond.after(node, 1);
  auto& traversal = graph.traverse();
  ASSERT_EQ(3, traversal.getClusters().size());
  {
    ASSERT_EQ(2, traversal.getClusters()[0].get<InputNode>().size());
    ASSERT_EQ(0, traversal.getClusters()[0].get<Node>().size());
    std::set<std::string> setNames, targetSetNames;
    setNames.insert(
        inner::getNodeTable(
            context)[traversal.getClusters()[0].get<InputNode>()[0].nodeIndex]
            ->name());
    setNames.insert(
        inner::getNodeTable(
            context)[traversal.getClusters()[0].get<InputNode>()[1].nodeIndex]
            ->name());
    targetSetNames.insert("Input1");
    targetSetNames.insert("Input2");
    ASSERT_EQ(setNames, targetSetNames);
    ASSERT_EQ(1, traversal.getClusters()[0].get<InputNode>()[0].output.size());
    ASSERT_EQ(1, traversal.getClusters()[0].get<InputNode>()[1].output.size());
    ASSERT_EQ(0, traversal.getClusters()[0].get<InputNode>()[0].input.size());
    ASSERT_EQ(0, traversal.getClusters()[0].get<InputNode>()[1].input.size());
  }
  {
    ASSERT_EQ(0, traversal.getClusters()[1].get<InputNode>().size());
    ASSERT_EQ(1, traversal.getClusters()[1].get<Node>().size());
    ASSERT_EQ(std::string("Node1"),
              inner::getNodeTable(
                  context)[traversal.getClusters()[1].get<Node>()[0].nodeIndex]
                  ->name());
    ASSERT_EQ(
        node_dyncast<Node*>(
            inner::getNodeTable(
                context)[traversal.getClusters()[1].get<Node>()[0].nodeIndex])
            ->getOperationPtr(),
        &operation);
    ASSERT_EQ(1, traversal.getClusters()[1].get<Node>()[0].output.size());
    ASSERT_EQ(1, traversal.getClusters()[1].get<Node>()[0].input.size());
  }
  {
    ASSERT_EQ(0, traversal.getClusters()[2].get<InputNode>().size());
    ASSERT_EQ(2, traversal.getClusters()[2].get<Node>().size());
    std::set<std::string> setNames, targetSetNames;
    setNames.insert(
        inner::getNodeTable(
            context)[traversal.getClusters()[2].get<Node>()[0].nodeIndex]
            ->name());
    setNames.insert(
        inner::getNodeTable(
            context)[traversal.getClusters()[2].get<Node>()[1].nodeIndex]
            ->name());
    targetSetNames.insert("Node2.1");
    targetSetNames.insert("Node2.2");
    ASSERT_EQ(setNames, targetSetNames);
    ASSERT_EQ(0, traversal.getClusters()[2].get<Node>()[0].output.size());
    ASSERT_EQ(0, traversal.getClusters()[2].get<Node>()[1].output.size());
    ASSERT_EQ(1, traversal.getClusters()[2].get<Node>()[0].input.size());
    ASSERT_EQ(1, traversal.getClusters()[2].get<Node>()[1].input.size());
    ASSERT_EQ(
        node_dyncast<Node*>(
            inner::getNodeTable(
                context)[traversal.getClusters()[2].get<Node>()[0].nodeIndex])
            ->getOperationPtr(),
        &operation);
    ASSERT_EQ(
        node_dyncast<Node*>(
            inner::getNodeTable(
                context)[traversal.getClusters()[2].get<Node>()[1].nodeIndex])
            ->getOperationPtr(),
        &operation);
  }
}
TEST_F(GraphTest, TraverseOnOtherGraph) {
  InputNode inputNodeFirst({1}, DataType::HALF, loader, context, false,
                           "Input1");
  InputNode inputNodeSecond({1}, DataType::HALF, loader, context, false,
                            "Input2");
  Node nodeFirst(operation, context, "Node1");
  Node nodeSecond(operation, context, "Node2");
  Node nodeOut(operation, context, "NodeOut");
  graph.addNode(inputNodeFirst);
  graph.addNode(inputNodeSecond);
  graph.addNode(nodeFirst);
  graph.addNode(nodeSecond);
  graph.addNode(nodeOut);
  inputNodeFirst.before(nodeFirst, 1);
  inputNodeFirst.before(nodeSecond, 2);
  inputNodeSecond.before(nodeFirst, 1);
  inputNodeSecond.before(nodeSecond, 2);
  nodeFirst.before(nodeOut, 1);
  nodeSecond.before(nodeOut, 2);
  Traversal traversal = graph.traverse();
  {
    ASSERT_EQ(2, traversal.getClusters()[0].get<InputNode>().size());
    ASSERT_EQ(0, traversal.getClusters()[0].get<Node>().size());
    std::set<std::string> setNames, targetSetNames;
    setNames.insert(
        inner::getNodeTable(
            context)[traversal.getClusters()[0].get<InputNode>()[0].nodeIndex]
            ->name());
    setNames.insert(
        inner::getNodeTable(
            context)[traversal.getClusters()[0].get<InputNode>()[1].nodeIndex]
            ->name());
    targetSetNames.insert("Input1");
    targetSetNames.insert("Input2");
    ASSERT_EQ(setNames, targetSetNames);
    ASSERT_EQ(2, traversal.getClusters()[0].get<InputNode>()[0].output.size());
    ASSERT_EQ(2, traversal.getClusters()[0].get<InputNode>()[1].output.size());
    ASSERT_EQ(0, traversal.getClusters()[0].get<InputNode>()[0].input.size());
    ASSERT_EQ(0, traversal.getClusters()[0].get<InputNode>()[1].input.size());
  }
  {
    ASSERT_EQ(0, traversal.getClusters()[1].get<InputNode>().size());
    ASSERT_EQ(2, traversal.getClusters()[1].get<Node>().size());
    std::set<std::string> setNames, targetSetNames;
    setNames.insert(
        inner::getNodeTable(
            context)[traversal.getClusters()[1].get<Node>()[0].nodeIndex]
            ->name());
    setNames.insert(
        inner::getNodeTable(
            context)[traversal.getClusters()[1].get<Node>()[1].nodeIndex]
            ->name());
    targetSetNames.insert("Node1");
    targetSetNames.insert("Node2");
    ASSERT_EQ(setNames, targetSetNames);
    ASSERT_EQ(1, traversal.getClusters()[1].get<Node>()[0].output.size());
    ASSERT_EQ(1, traversal.getClusters()[1].get<Node>()[1].output.size());
    ASSERT_EQ(1, traversal.getClusters()[1].get<Node>()[0].input.size());
    ASSERT_EQ(1, traversal.getClusters()[1].get<Node>()[1].input.size());
  }
  {
    ASSERT_EQ(0, traversal.getClusters()[2].get<InputNode>().size());
    ASSERT_EQ(1, traversal.getClusters()[2].get<Node>().size());
    ASSERT_EQ(std::string("NodeOut"),
              inner::getNodeTable(
                  context)[traversal.getClusters()[2].get<Node>()[0].nodeIndex]
                  ->name());
    ASSERT_EQ(0, traversal.getClusters()[2].get<Node>()[0].output.size());
    ASSERT_EQ(2, traversal.getClusters()[2].get<Node>()[0].input.size());
  }
}
} // namespace athena::core
