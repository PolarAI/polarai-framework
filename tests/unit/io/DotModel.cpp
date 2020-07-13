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
#include <polarai/core/node/OutputNode.hpp>
#include <polarai/io/DotModel.hpp>
#include <polarai/operation/AddOperation.hpp>
#include <polarai/operation/LogLossOperation.hpp>
#include <polarai/operation/MatMulOperation.hpp>
#include <polarai/operation/SigmoidOperation.hpp>

#include <gtest/gtest.h>

using namespace polarai;
using namespace polarai::core;
using namespace polarai::operation;

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
  auto [graphGradient, graphConnector] = graph.getGradient(node);
  std::cout << "*** Function: ***" << std::endl;
  io::DotModel::exportGraph(graph, std::cout);
  std::cout << "*** Gradient: ***" << std::endl;
  io::DotModel::exportGraph(graphGradient, std::cout);
  std::cout << "*** Connector: ***" << std::endl;
  io::DotModel::exportGraph(graphConnector, std::cout);
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
  auto [graphGradient, graphConnector] = graph.getGradient(node2);
  std::cout << "*** Function: ***" << std::endl;
  io::DotModel::exportGraph(graph, std::cout);
  std::cout << "*** Gradient: ***" << std::endl;
  io::DotModel::exportGraph(graphGradient, std::cout);
  std::cout << "*** Connector: ***" << std::endl;
  io::DotModel::exportGraph(graphConnector, std::cout);
}

TEST(DotModel, TopologyLogReg) {
  Context context;
  auto graph = context.create<Graph>("graph1");
  size_t size = 3;
  auto inpVector = graph.create<InputNode>(TensorShape{1, size}, DataType::FLOAT, true,
                                      0, "inpVector");
  auto weightsVector = graph.create<InputNode>(TensorShape{size, 1}, DataType::FLOAT, false,
                                      0, "weightsVector");
  auto operationMatMulId = context.create<MatMulOperation>(false, false, "gemm");
  auto nodeMatMul = graph.create<Node>(operationMatMulId, "nodeGemm");
  graph.connect(inpVector, nodeMatMul, MatMulOperation::LEFT);
  graph.connect(weightsVector, nodeMatMul, MatMulOperation::RIGHT);
  auto operationSigmoidId = context.create<SigmoidOperation>("gemm");
  auto nodeSigmoid = graph.create<Node>(operationSigmoidId, "nodeSigmoid");
  graph.connect(nodeMatMul, nodeSigmoid, SigmoidOperation::Unmarked);
  auto operationLogLossId = context.create<LogLossOperation>("logloss");
  auto loss = graph.create<Node>(operationLogLossId, "loss");
  auto inpGroundTruth = graph.create<InputNode>(TensorShape{1, 1}, DataType::FLOAT, true,
                                                0, "groundTruth");
  graph.connect(nodeSigmoid, loss, LogLossOperation::PREDICTED);
  graph.connect(inpGroundTruth, loss, LogLossOperation::GROUND_TRUTH);
  auto [graphGradient, graphConnector] = graph.getGradient(loss);
  std::cout << "*** Function: ***" << std::endl;
  io::DotModel::exportGraph(graph, std::cout);
  std::cout << "*** Gradient: ***" << std::endl;
  io::DotModel::exportGraph(graphGradient, std::cout);
  std::cout << "*** Connector: ***" << std::endl;
  io::DotModel::exportGraph(graphConnector, std::cout);
}
} // namespace
