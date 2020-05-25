#include <athena/core/context/Context.h>
#include <athena/core/graph/Graph.h>
#include <athena/core/node/InputNode.h>
#include <athena/core/node/Node.h>
#include <athena/core/node/OutputNode.h>
#include <athena/model/DotModel.h>
#include <athena/operation/AddOperation.h>
#include <gtest/gtest.h>

using namespace athena;
using namespace athena::core;
using namespace athena::operation;

namespace {
TEST(DotModel, Topology1) {
  Context context;
  auto graph = context.create<Graph>("mygraph");
  auto inp1 = graph.create<InputNode>(TensorShape{2, 2}, DataType::FLOAT, true,
                                      0, "inp1");
  auto inp2 = graph.create<InputNode>(TensorShape{2, 2}, DataType::FLOAT, false,
                                      0, "inp2");
  auto operationId = context.create<AddOperation>("myop");
  auto node = graph.create<Node>(operationId, "mynode");
  graph.connect(inp1, node, AddOperation::LEFT);
  graph.connect(inp2, node, AddOperation::RIGHT);
  auto out = graph.create<OutputNode>("out");
  graph.connect(node, out, Operation::Unmarked);
  auto [graphGradient, graphConnector] = graph.getGradient();
  std::cout << "*** Function: ***" << std::endl;
  model::DotModel::exportGraph(graph, std::cout);
  std::cout << "*** Gradient: ***" << std::endl;
  model::DotModel::exportGraph(graphGradient, std::cout);
  std::cout << "*** Connector: ***" << std::endl;
  model::DotModel::exportGraph(graphConnector, std::cout);
}

TEST(DotModel, Topology2) {
  Context context;
  auto graph = context.create<Graph>("graph1");
  auto inp1 = graph.create<InputNode>(TensorShape{2, 2}, DataType::FLOAT, false,
                                      0, "inp1");
  auto inp2 = graph.create<InputNode>(TensorShape{2, 2}, DataType::FLOAT, true,
                                      0, "inp2");
  auto operationId = context.create<AddOperation>("add_op");
  auto node1 = graph.create<Node>(operationId, "node1");
  graph.connect(inp1, node1, AddOperation::LEFT);
  graph.connect(inp2, node1, AddOperation::RIGHT);
  auto inp3 = graph.create<InputNode>(TensorShape{2, 2}, DataType::FLOAT, false,
                                      0, "inp3");
  auto node2 = graph.create<Node>(operationId, "node2");
  graph.connect(inp3, node2, AddOperation::LEFT);
  graph.connect(node1, node2, AddOperation::RIGHT);
  auto out = graph.create<OutputNode>("out");
  graph.connect(node2, out, Operation::Unmarked);
  auto [graphGradient, graphConnector] = graph.getGradient();
  std::cout << "*** Function: ***" << std::endl;
  model::DotModel::exportGraph(graph, std::cout);
  std::cout << "*** Gradient: ***" << std::endl;
  model::DotModel::exportGraph(graphGradient, std::cout);
  std::cout << "*** Connector: ***" << std::endl;
  model::DotModel::exportGraph(graphConnector, std::cout);
}
} // namespace
