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

#include <athena/core/GraphCompiler.h>
#include <athena/core/InputNode.h>
#include <athena/core/LossNode.h>
#include <athena/core/Node.h>
#include <athena/core/Graph.h>

using namespace athena::core::inner;

namespace athena::core {

static void compileInputNodes(
    AbstractGenerator &generator,
    const GraphCompiler::ClusterContainer<core::InputNode> &inputNodes,
    athena::core::Graph &graph) {
    for (auto &nodeDeps : inputNodes) {
        auto &inputNode =
            node_cast<core::InputNode &>(*core::inner::getNodeTable(
                core::inner::getContext(graph))[nodeDeps.nodeIndex]);
        generator.openNode(inputNode.getName());
        generator.generate("allocate",
                           core::inner::getTensorFromNode(inputNode));
        generator.generateLoad(inputNode.getLoader(),
                               core::inner::getTensorFromNode(inputNode));
        generator.closeNode();
    }
}
static void compileActionNodes(
    AbstractGenerator &generator,
    const GraphCompiler::ClusterContainer<core::Node> &actionNodes,
    athena::core::Graph &graph) {
    for (auto &nodeDeps : actionNodes) {
        //std::vector<core::inner::Tensor *> preparedTensors;
//        for (auto &input : nodeDeps.input) {
//            auto *node = core::inner::getNodeTable(
//                core::inner::getContext(graph))[input.nodeIndex];
//            preparedTensors.push_back(&core::inner::getTensorFromNode(*node));
//        }
        auto inputs = getOperationArgs(core::inner::getContext(graph), nodeDeps);
        auto &node = node_cast<core::Node &>(*core::inner::getNodeTable(
            core::inner::getContext(graph))[nodeDeps.nodeIndex]);
        generator.openNode(node.getName());
        generator.generate("allocate", core::inner::getTensorFromNode(node));
        //preparedTensors.push_back(&core::inner::getTensorFromNode(node));
        // todo lock tensors in memory
        node.getOperation().gen(generator, inputs);
        // todo unlock tensors in memory

        generator.closeNode();
    }
}
static void compileLossNodes(
    AbstractGenerator &generator,
    const GraphCompiler::ClusterContainer<core::LossNode> &lossNodes,
    athena::core::Graph &graph) {
    for (auto &nodeDeps : lossNodes) {
        auto inputs =
            getOperationArgs(core::inner::getContext(graph), nodeDeps);
        auto &node =
            *reinterpret_cast<core::LossNode *>(core::inner::getNodeTable(
                core::inner::getContext(graph))[nodeDeps.nodeIndex]);
        generator.openNode(node.getName());
        generator.generate("allocate", core::inner::getTensorFromNode(node));
        // todo lock tensors in memory
        node.getOperation().gen(generator, inputs);
        // todo unlock tensors in memory
        generator.closeNode();
    }
}

static void compileLossDerivatives(
    AbstractGenerator &generator,
    const GraphCompiler::ClusterContainer<LossNode> &lossNodes,
    Optimizer &graphOptimizer,
    Graph &graph) {
    for (auto &nodeDeps : lossNodes) {
        auto operationArgs = getOperationArgs(getContext(graph), nodeDeps);
        auto &lossNode = node_cast<core::LossNode &>(*getNodeTable(
            getContext(graph))[nodeDeps.nodeIndex]);
        generator.generate("allocate", getOwnDerivative(lossNode));
        lossNode.getOperation().genOwnDerivative(generator, getOutgoingDerivatives(lossNode),
            getOwnDerivative(lossNode));
        for (auto& inputNode : nodeDeps.input) {
            auto inputNodeIndex = inputNode.second;
            auto inputNodeMark = inputNode.first;
            auto &derivativeTensor = getOutgoingDerivative(*getNodeTable(getContext(graph))[inputNodeIndex], lossNode.getNodeIndex());
            generator.generate("allocate", derivativeTensor);
            // todo lock tensors in memory
            lossNode.getOperation().genIncomingDerivative(generator, operationArgs, derivativeTensor, getOwnDerivative(lossNode), inputNodeMark);
            // TODO memory clean up
        }
    }
}

static void compileNodeDerivatives(
    AbstractGenerator &generator,
    const GraphCompiler::ClusterContainer<core::Node> &nodes,
    core::Optimizer &graphOptimizer,
    core::Graph &graph) {
    for (auto &nodeDeps : nodes) {
//        std::vector<core::inner::Tensor *> inputs;
//        for (auto &inp : nodeDeps.input) {
//            auto &tensor =
//                core::inner::getTensorFromNode(*core::inner::getNodeTable(
//                    core::inner::getContext(graph))[inp.nodeIndex]);
//
//            inputs.push_back(&tensor);
//        }
        auto operationArgs = getOperationArgs(getContext(graph), nodeDeps);

        auto &node = node_cast<core::Node &>(*core::inner::getNodeTable(
            core::inner::getContext(graph))[nodeDeps.nodeIndex]);
        generator.generate("allocate", getOwnDerivative(node));
        node.getOperation().genOwnDerivative(generator, getOutgoingDerivatives(node),
            getOwnDerivative(node));
        for (auto& inputNode : nodeDeps.input) {
            //auto &derivativeTensor = getOutgoingDerivative(lossNode, idx);
            auto inputNodeIndex = inputNode.second;
            auto inputNodeMark = inputNode.first;
            auto &derivativeTensor = getOutgoingDerivative(*getNodeTable(getContext(graph))[inputNodeIndex], node.getNodeIndex());
            generator.generate("allocate", derivativeTensor);
            // todo lock tensors in memory
            node.getOperation().genIncomingDerivative(generator, operationArgs, derivativeTensor, getOwnDerivative(node), inputNodeMark);
            // TODO memory clean up
        }
    }
}

//static void adjustWeights(
//    AbstractGenerator &generator,
//    const GraphCompiler::ClusterContainer<core::InputNode> &inputNodes,
//    core::Optimizer &graphOptimizer,
//    core::Graph &graph) {
//    for (auto &nodeDeps : inputNodes) {
//        auto &inputNode =
//            node_cast<core::InputNode &>(*core::inner::getNodeTable(
//                core::inner::getContext(graph))[nodeDeps.nodeIndex]);
//
//        // Frozen nodes are usually user data thus not updated
//        if (inputNode.isFrozen()) continue;
//
//        // todo lock in memory
//        auto &tensor = core::inner::getTensorFromNode(inputNode);
//
//        std::vector<core::inner::Tensor *> incomingErrors;
//        for (auto &outp : nodeDeps.output) {
//            auto &abstractNode = *core::inner::getNodeTable(
//                core::inner::getContext(graph))[outp.nodeIndex];
//            if (abstractNode.getType() == core::NodeType::LOSS ||
//                abstractNode.getType() == core::NodeType::DEFAULT) {
//                auto &outpNode = *reinterpret_cast<core::Node *>(&abstractNode);
//                auto &derivativeTensor =
//                    core::inner::getOutgoingDerivative(outpNode, outp.mark - 1);
//                incomingErrors.push_back(&derivativeTensor);
//            }
//        }
//
//        // Apply error correction
//        graphOptimizer.genFix(generator, tensor, incomingErrors);
//    }
//}

static void compileDerivatives(AbstractGenerator &generator,
                               const core::Traversal &traversal,
                               core::Optimizer &graphOptimizer,
                               core::Graph &graph) {
    auto clusters = traversal.getClusters();

    for (auto clusterIt = clusters.rbegin(); clusterIt != clusters.rend();
         ++clusterIt) {
        auto &lossNodes = clusterIt->get<core::LossNode>();
        compileLossDerivatives(generator, lossNodes, graphOptimizer, graph);

        auto &actionNodes = clusterIt->get<core::Node>();
        compileNodeDerivatives(generator, actionNodes, graphOptimizer, graph);

        auto &inputNodes = clusterIt->get<core::InputNode>();
        //adjustWeights(generator, inputNodes, graphOptimizer, graph);
    }
}

void GraphCompiler::compileForward(athena::core::Graph &graph,
                                   athena::core::AbstractGenerator &generator) {
    generator.generateFunctionHeader("evaluateGraph");

    auto graphTraversal = graph.traverse();

    for (auto &cluster : graphTraversal.getClusters()) {
        auto &inputNodes = cluster.get<core::InputNode>();
        compileInputNodes(generator, inputNodes, graph);

        auto &actionNodes = cluster.get<core::Node>();
        compileActionNodes(generator, actionNodes, graph);

        auto &lossNodes = cluster.get<core::LossNode>();
        compileLossNodes(generator, lossNodes, graph);
    }

    generator.generateFunctionFooter();
}
void GraphCompiler::compileBackward(Graph &graph,
                                    AbstractGenerator &generator) {
    auto graphTraversal = graph.traverse();
    generator.generateFunctionHeader("optimizeGraph");
    compileDerivatives(generator, graphTraversal, *graph.getOptimizer(), graph);
    generator.generateFunctionFooter();
}
}  // namespace athena::core
