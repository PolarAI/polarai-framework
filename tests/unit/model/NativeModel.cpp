/*
 * Copyright (c) 2019 Athena. All rights reserved.
 * https://getathena.ml
 *
 * Licensed under MIT license.
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS, WITHOUT
 * WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the
 * License for the specific language governing permissions and limitations under
 * the License.
 */
//
//#include <athena/core/Criterion.h>
//#include <athena/core/LossNode.h>
//#include <athena/core/context/Context.h>
//#include <athena/core/graph/Graph.h>
//#include <athena/model/NativeModel.h>
//
//#include <gtest/gtest.h>
//#include <sstream>
//
// using namespace athena::core;
// using namespace athena::model;
//
// static bool findNodeByNameInGraph(Graph& graph, const std::string& name) {
//  auto& owningStorage = graph.getOwningStorage();
//  auto& inputNodes = std::get<std::vector<InputNodeImpl>>(owningStorage);
//  for (auto& node : inputNodes) {
//    if (node.getName() == name) {
//      return true;
//    }
//  }
//  auto& actionNodes = std::get<std::vector<Node>>(owningStorage);
//  for (auto& node : actionNodes) {
//    if (node.getName() == name) {
//      return true;
//    }
//  }
//  auto& lossNodes = std::get<std::vector<LossNode>>(owningStorage);
//  for (auto& node : lossNodes) {
//    if (node.getName() == name) {
//      return true;
//    }
//  }
//  auto& outputNodes =
//  std::get<std::vector<OutputNodeInternal>>(owningStorage); for (auto& node :
//  outputNodes) {
//    if (node.getName() == name) {
//      return true;
//    }
//  }
//  return false;
//}
//
// TEST(ModelTest, NativeModelSmokeTest) {
//  // Arrange
//  Context ctx;
//  Graph graph(ctx);
//
//  DummyLoader dummyLoader;
//  InputNodeImpl input({0}, DataType::UNDEFINED, dummyLoader, ctx, true,
//  "input"); graph.addNode(input);
//
//  OperationDummy dummyOp("dummy");
//  Node node(dummyOp, ctx, "node");
//  graph.addNode(node);
//  node.after(input, 1);
//
//  LossNode loss(dummyOp, Criterion::UNDEFINED, ctx, "loss");
//  graph.addNode(loss);
//  loss.after(node, 1);
//
//  OutputNodeInternal output(DataType::UNDEFINED, ctx, "output");
//  graph.addNode(output);
//  output.after(loss, 1);
//
//  // Act
//  NativeModel::saveGraphToFile(graph, "file.model");
//  Graph newGraph(ctx);
//  NativeModel::readGraphFromFile(newGraph, "file.model");
//
//  // Assert
//  EXPECT_TRUE(findNodeByNameInGraph(newGraph, "input"));
//  EXPECT_TRUE(findNodeByNameInGraph(newGraph, "node"));
//  EXPECT_TRUE(findNodeByNameInGraph(newGraph, "loss"));
//  EXPECT_TRUE(findNodeByNameInGraph(newGraph, "output"));
//}