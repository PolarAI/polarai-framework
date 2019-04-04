/*
 * Copyright (c) 2018 Athena. All rights reserved.
 * https://athenaframework.ml
 *
 * Licensed under MIT license.
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an “AS IS” BASIS, WITHOUT
 * WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the
 * License for the specific language governing permissions and limitations under
 * the License.
 */

#include <athena/core/Graph.h>
#include <athena/core/inner/GlobalTables.h>
#include <set>

#include <gtest/gtest.h>

namespace athena::core {
TEST(GraphTest, Creation) {
    Graph graph;
}
TEST(GraphTest, Using) {
    clearAll();
    Graph graph;
    DummyLoader loader;
    OperationDummy operation("DummyOperation");
    InputNode inputNodeFirst({1}, DataType::HALF, loader, "Input1");
    ASSERT_EQ(inputNodeFirst.getNodeIndex(), 1);
    graph.addNode(inputNodeFirst);
    InputNode inputNodeSecond({1}, DataType::HALF, loader, "Input2");
    ASSERT_EQ(inputNodeSecond.getNodeIndex(), 2);
    graph.addNode(inputNodeSecond);
    Node node({1}, DataType::HALF, operation, "Node1");
    ASSERT_EQ(node.getNodeIndex(), 3);
    graph.addNode(node);
    inputNodeFirst.before(node, 1);
    inputNodeSecond.before(node, 1);
    Node nodeSecondLayerFirst({1}, DataType::FLOAT, operation, "Node2.1");
    ASSERT_EQ(nodeSecondLayerFirst.getNodeIndex(), 4);
    Node nodeSecondLayerSecond({1}, DataType::FLOAT, operation, "Node2.2");
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
        {node.getNodeIndex(), nodeSecondLayerSecond.getNodeIndex(), 1}
    };
    auto topology = graph.getTopology();
    std::sort(required_topology.begin(), required_topology.end());
    std::sort(topology.begin(), topology.end());
    ASSERT_EQ(required_topology, topology);
    node.saveInGraph();
    ASSERT_EQ(node.getNodeIndex(), 6);
    ASSERT_EQ(graph.countOwningNodes(), 1);
    ASSERT_EQ(graph.countSyncNodes(), 4);
    ASSERT_EQ(required_topology, graph.getTopology());
}
TEST(GraphTest, RemoveNode) {
    clearAll();
    Graph graph;
    DummyLoader loader;
    OperationDummy operation("DummyOperation");
    InputNode inputNodeFirst({1}, DataType::HALF, loader, "Input1");
    ASSERT_EQ(inputNodeFirst.getNodeIndex(), 1);
    graph.addNode(inputNodeFirst);
    InputNode inputNodeSecond({1}, DataType::HALF, loader, "Input2");
    ASSERT_EQ(inputNodeSecond.getNodeIndex(), 2);
    graph.addNode(inputNodeSecond);
    Node node({1}, DataType::HALF, operation, "Node1");
    ASSERT_EQ(node.getNodeIndex(), 3);
    graph.addNode(node);
    inputNodeFirst.before(node, 1);
    inputNodeSecond.before(node, 1);
    Node nodeSecondLayerFirst({1}, DataType::FLOAT, operation, "Node2.1");
    ASSERT_EQ(nodeSecondLayerFirst.getNodeIndex(), 4);
    Node nodeSecondLayerSecond({1}, DataType::FLOAT, operation, "Node2.2");
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
TEST(GraphTest, DeepTestTraverse) {
    clearAll();
    Graph graph;
    DummyLoader loader;
    OperationDummy operation("DummyOperation");
    InputNode inputNodeFirst({1}, DataType::HALF, loader, "Input1");
    ASSERT_EQ(inputNodeFirst.getNodeIndex(), 1);
    graph.addNode(inputNodeFirst);
    InputNode inputNodeSecond({1}, DataType::HALF, loader, "Input2");
    ASSERT_EQ(inputNodeSecond.getNodeIndex(), 2);
    graph.addNode(inputNodeSecond);
    Node node({1}, DataType::HALF, operation, "Node1");
    ASSERT_EQ(node.getNodeIndex(), 3);
    graph.addNode(node);
    inputNodeFirst.before(node, 1);
    inputNodeSecond.before(node, 1);
    Node nodeSecondLayerFirst({1}, DataType::FLOAT, operation, "Node2.1");
    ASSERT_EQ(nodeSecondLayerFirst.getNodeIndex(), 4);
    Node nodeSecondLayerSecond({1}, DataType::FLOAT, operation, "Node2.2");
    ASSERT_EQ(nodeSecondLayerSecond.getNodeIndex(), 5);
    graph.addNode(nodeSecondLayerFirst);
    graph.addNode(nodeSecondLayerSecond);
    nodeSecondLayerFirst.after(node, 1);
    nodeSecondLayerSecond.after(node, 1);
    Traversal traversal = graph.traverse(false);
    ASSERT_EQ(3, traversal.clusters.size());
    {
        ASSERT_EQ(2, traversal.clusters[0].get<InputNode>().size());
        ASSERT_EQ(0, traversal.clusters[0].get<Node>().size());
        std::set<std::string> setNames, targetSetNames;
        setNames.insert(traversal.clusters[0].get<InputNode>()[0].node.name());
        setNames.insert(traversal.clusters[0].get<InputNode>()[1].node.name());
        targetSetNames.insert("Input1");
        targetSetNames.insert("Input2");
        ASSERT_EQ(setNames, targetSetNames);
        ASSERT_EQ(1, traversal.clusters[0].get<InputNode>()[0].output.size());
        ASSERT_EQ(1, traversal.clusters[0].get<InputNode>()[1].output.size());
        ASSERT_EQ(0, traversal.clusters[0].get<InputNode>()[0].input.size());
        ASSERT_EQ(0, traversal.clusters[0].get<InputNode>()[1].input.size());
    }
    {
        ASSERT_EQ(0, traversal.clusters[1].get<InputNode>().size());
        ASSERT_EQ(1, traversal.clusters[1].get<Node>().size());
        ASSERT_EQ(std::string("Node1"),
                  traversal.clusters[1].get<Node>()[0].node.name());
        ASSERT_EQ(2, traversal.clusters[1].get<Node>()[0].output.size());
        ASSERT_EQ(2, traversal.clusters[1].get<Node>()[0].input.size());
    }
    {
        ASSERT_EQ(0, traversal.clusters[2].get<InputNode>().size());
        ASSERT_EQ(2, traversal.clusters[2].get<Node>().size());
        std::set<std::string> setNames, targetSetNames;
        setNames.insert(traversal.clusters[2].get<Node>()[0].node.name());
        setNames.insert(traversal.clusters[2].get<Node>()[1].node.name());
        targetSetNames.insert("Node2.1");
        targetSetNames.insert("Node2.2");
        ASSERT_EQ(setNames, targetSetNames);
        ASSERT_EQ(0, traversal.clusters[2].get<Node>()[0].output.size());
        ASSERT_EQ(0, traversal.clusters[2].get<Node>()[1].output.size());
        ASSERT_EQ(1, traversal.clusters[2].get<Node>()[0].input.size());
        ASSERT_EQ(1, traversal.clusters[2].get<Node>()[1].input.size());
    }
}
TEST(GraphTest, TraverseOnOtherGraph) {
    clearAll();
    Graph graph;
    DummyLoader loader;
    OperationDummy operation("DummyOperation");
    InputNode inputNodeFirst({1}, DataType::HALF, loader, "Input1");
    InputNode inputNodeSecond({1}, DataType::HALF, loader, "Input2");
    Node nodeFirst({1}, DataType::FLOAT, operation, "Node1");
    Node nodeSecond({1}, DataType::FLOAT, operation, "Node2");
    Node nodeOut({1}, DataType::HALF, operation, "NodeOut");
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
    Traversal traversal = graph.traverse(false);
    {
        ASSERT_EQ(2, traversal.clusters[0].get<InputNode>().size());
        ASSERT_EQ(0, traversal.clusters[0].get<Node>().size());
        std::set<std::string> setNames, targetSetNames;
        setNames.insert(traversal.clusters[0].get<InputNode>()[0].node.name());
        setNames.insert(traversal.clusters[0].get<InputNode>()[1].node.name());
        targetSetNames.insert("Input1");
        targetSetNames.insert("Input2");
        ASSERT_EQ(setNames, targetSetNames);
        ASSERT_EQ(2, traversal.clusters[0].get<InputNode>()[0].output.size());
        ASSERT_EQ(2, traversal.clusters[0].get<InputNode>()[1].output.size());
        ASSERT_EQ(0, traversal.clusters[0].get<InputNode>()[0].input.size());
        ASSERT_EQ(0, traversal.clusters[0].get<InputNode>()[1].input.size());
    }
    {
        ASSERT_EQ(0, traversal.clusters[1].get<InputNode>().size());
        ASSERT_EQ(2, traversal.clusters[1].get<Node>().size());
        std::set<std::string> setNames, targetSetNames;
        setNames.insert(traversal.clusters[1].get<Node>()[0].node.name());
        setNames.insert(traversal.clusters[1].get<Node>()[1].node.name());
        targetSetNames.insert("Node1");
        targetSetNames.insert("Node2");
        ASSERT_EQ(setNames, targetSetNames);
        ASSERT_EQ(1, traversal.clusters[1].get<Node>()[0].output.size());
        ASSERT_EQ(1, traversal.clusters[1].get<Node>()[1].output.size());
        ASSERT_EQ(2, traversal.clusters[1].get<Node>()[0].input.size());
        ASSERT_EQ(2, traversal.clusters[1].get<Node>()[1].input.size());
    }
    {
        ASSERT_EQ(0, traversal.clusters[2].get<InputNode>().size());
        ASSERT_EQ(1, traversal.clusters[2].get<Node>().size());
        ASSERT_EQ(std::string("NodeOut"),
                  traversal.clusters[2].get<Node>()[0].node.name());
        ASSERT_EQ(0, traversal.clusters[2].get<Node>()[0].output.size());
        ASSERT_EQ(2, traversal.clusters[2].get<Node>()[0].input.size());
    }
}
}
