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

namespace athena::core {

static void compileInputNodes(
    AbstractGenerator& generator,
    const GraphCompiler::ClusterContainer<core::InputNode>& inputNodes,
    athena::core::Graph& graph) {
  for (auto& nodeDeps : inputNodes) {
    auto& inputNode = node_cast<core::InputNode&>(*core::inner::getNodeTable(
        core::inner::getContext(graph))[nodeDeps.nodeIndex]);
    generator.openNode(inputNode.getName());
    generator.generate("allocate", core::inner::getTensorFromNode(inputNode));
    generator.generateLoad(inputNode.getLoader(),
                           core::inner::getTensorFromNode(inputNode));
    generator.closeNode();
  }
}
static void compileActionNodes(
    AbstractGenerator& generator,
    const GraphCompiler::ClusterContainer<core::Node>& actionNodes,
    athena::core::Graph& graph) {
  for (auto& nodeDeps : actionNodes) {
    std::vector<core::inner::Tensor*> preparedTensors;
    for (auto& input : nodeDeps.input) {
      auto* node = core::inner::getNodeTable(
          core::inner::getContext(graph))[input.nodeIndex];
      preparedTensors.push_back(&core::inner::getTensorFromNode(*node));
    }
    auto& node = node_cast<core::Node&>(*core::inner::getNodeTable(
        core::inner::getContext(graph))[nodeDeps.nodeIndex]);
    generator.openNode(node.getName());
    generator.generate("allocate", core::inner::getTensorFromNode(node));
    preparedTensors.push_back(&core::inner::getTensorFromNode(node));
    // todo lock tensors in memory
    node.getOperation().gen(generator, preparedTensors);
    // todo unlock tensors in memory

    generator.closeNode();
  }
}
static void compileLossNodes(
    AbstractGenerator& generator,
    const GraphCompiler::ClusterContainer<core::LossNode>& lossNodes,
    athena::core::Graph& graph) {
  if (lossNodes.size() == 1) {
    auto& nodeDeps = lossNodes[0];
    std::vector<core::inner::Tensor*> preparedTensors;
    for (auto& input : nodeDeps.input) {
      auto* node = core::inner::getNodeTable(
          core::inner::getContext(graph))[input.nodeIndex];
      preparedTensors.push_back(&core::inner::getTensorFromNode(*node));
    }
    auto& node = *reinterpret_cast<core::LossNode*>(core::inner::getNodeTable(
        core::inner::getContext(graph))[nodeDeps.nodeIndex]);
    generator.openNode(node.getName());
    generator.generate("allocate", core::inner::getTensorFromNode(node));
    preparedTensors.push_back(&core::inner::getTensorFromNode(node));
    // todo lock tensors in memory
    node.getOperation().gen(generator, preparedTensors);
    // todo unlock tensors in memory

    generator.closeNode();
  } else if (lossNodes.size() > 1) {
    new core::FatalError(ATH_FATAL_OTHER, "More than 1 loss node");
  }
}

static void compileLossDerivatives(
    AbstractGenerator& generator,
    const GraphCompiler::ClusterContainer<core::LossNode>& lossNodes,
    core::Optimizer& graphOptimizer, core::Graph& graph) {
  for (auto& nodeDeps : lossNodes) {
    // Collect inputs
    std::vector<core::inner::Tensor*> inputs;
    for (auto& inp : nodeDeps.input) {
      auto& tensor = core::inner::getTensorFromNode(*core::inner::getNodeTable(
          core::inner::getContext(graph))[inp.nodeIndex]);

      inputs.push_back(&tensor);
    }

    auto& lossNode = node_cast<core::LossNode&>(*core::inner::getNodeTable(
        core::inner::getContext(graph))[nodeDeps.nodeIndex]);

    auto& outputTensor = core::inner::getTensorFromNode(lossNode);

    for (size_t idx = 0; idx < lossNode.getOperation().getOperandsCount() - 1;
         idx++) {
      auto& derivativeTensor = core::inner::getDerivativeTensor(lossNode, idx);

      generator.generate("allocate", derivativeTensor);
      // todo lock tensors in memory
      lossNode.getOperation().genDerivative(
          graphOptimizer.getRequiredOrder(), generator, outputTensor,
          *core::inner::getNullTensor(
              core::inner::getContext(graph)), // todo re-think this
          inputs, derivativeTensor, idx);
      // TODO memory clean up
    }
  }
}

static void
compileNodeDerivatives(AbstractGenerator& generator,
                       const GraphCompiler::ClusterContainer<core::Node>& nodes,
                       core::Optimizer& graphOptimizer, core::Graph& graph) {
  for (auto& nodeDeps : nodes) {
    std::vector<core::inner::Tensor*> inputs;
    for (auto& inp : nodeDeps.input) {
      auto& tensor = core::inner::getTensorFromNode(*core::inner::getNodeTable(
          core::inner::getContext(graph))[inp.nodeIndex]);

      inputs.push_back(&tensor);
    }

    auto& node = node_cast<core::Node&>(*core::inner::getNodeTable(
        core::inner::getContext(graph))[nodeDeps.nodeIndex]);

    // Calculate total error before this node
    std::vector<core::inner::Tensor*> incomingErrors;
    for (auto& outp : nodeDeps.output) {
      auto& abstractNode = *core::inner::getNodeTable(
          core::inner::getContext(graph))[outp.nodeIndex];
      if (abstractNode.getType() == core::NodeType::LOSS ||
          abstractNode.getType() == core::NodeType::DEFAULT) {
        auto& outpNode = *reinterpret_cast<core::Node*>(&abstractNode);
        auto& tensor =
            core::inner::getDerivativeTensor(outpNode, outp.mark - 1);
        incomingErrors.push_back(&tensor);
      }
    }

    auto& errorTensor = core::inner::getIncomingGradient(node);
    generator.generate("allocate", errorTensor);
    graphOptimizer.genError(generator, incomingErrors, errorTensor);

    auto& outputTensor = core::inner::getTensorFromNode(node);

    std::vector<core::inner::Tensor*> derivativeTensors;

    for (size_t idx = 0; idx < node.getOperation().getOperandsCount(); idx++) {
      auto& derivativeTensor = core::inner::getDerivativeTensor(node, idx);

      derivativeTensors.push_back(&derivativeTensor);

      generator.generate("allocate", derivativeTensor);
      // todo lock tensors in memory
      node.getOperation().genDerivative(graphOptimizer.getRequiredOrder(),
                                        generator, outputTensor, errorTensor,
                                        inputs, derivativeTensor, idx);
      // TODO memory clean up
    }

    std::vector<core::inner::Tensor*> internalErrors;

    for (size_t idx = 0; idx < node.getOperation().getOperandsCount(); idx++) {
      internalErrors.push_back(&errorTensor);

      generator.generate("allocate", errorTensor);
    }
  }
}

static void adjustWeights(
    AbstractGenerator& generator,
    const GraphCompiler::ClusterContainer<core::InputNode>& inputNodes,
    core::Optimizer& graphOptimizer, core::Graph& graph) {
  for (auto& nodeDeps : inputNodes) {
    auto& inputNode = node_cast<core::InputNode&>(*core::inner::getNodeTable(
        core::inner::getContext(graph))[nodeDeps.nodeIndex]);

    // Frozen nodes are usually user data thus not updated
    if (inputNode.isFrozen())
      continue;

    // todo lock in memory
    auto& tensor = core::inner::getTensorFromNode(inputNode);

    std::vector<core::inner::Tensor*> incomingErrors;
    for (auto& outp : nodeDeps.output) {
      auto& abstractNode = *core::inner::getNodeTable(
          core::inner::getContext(graph))[outp.nodeIndex];
      if (abstractNode.getType() == core::NodeType::LOSS ||
          abstractNode.getType() == core::NodeType::DEFAULT) {
        auto& outpNode = *reinterpret_cast<core::Node*>(&abstractNode);
        auto& derivativeTensor =
            core::inner::getDerivativeTensor(outpNode, outp.mark - 1);
        incomingErrors.push_back(&derivativeTensor);
      }
    }

    // Apply error correction
    graphOptimizer.genFix(generator, tensor, incomingErrors);
  }
}

static void compileDerivatives(AbstractGenerator& generator,
                               const core::Traversal& traversal,
                               core::Optimizer& graphOptimizer,
                               core::Graph& graph) {
  auto clusters = traversal.getClusters();

  for (auto clusterIt = clusters.rbegin(); clusterIt != clusters.rend();
       ++clusterIt) {
    auto& lossNodes = clusterIt->get<core::LossNode>();
    compileLossDerivatives(generator, lossNodes, graphOptimizer, graph);

    auto& actionNodes = clusterIt->get<core::Node>();
    compileNodeDerivatives(generator, actionNodes, graphOptimizer, graph);

    auto& inputNodes = clusterIt->get<core::InputNode>();
    adjustWeights(generator, inputNodes, graphOptimizer, graph);
  }
}

void GraphCompiler::compileForward(athena::core::Graph& graph,
                                   athena::core::AbstractGenerator& generator) {
  generator.generateFunctionHeader("evaluateGraph");

  auto graphTraversal = graph.traverse();

  for (auto& cluster : graphTraversal.getClusters()) {
    auto& inputNodes = cluster.get<core::InputNode>();
    compileInputNodes(generator, inputNodes, graph);

    auto& actionNodes = cluster.get<core::Node>();
    compileActionNodes(generator, actionNodes, graph);

    auto& lossNodes = cluster.get<core::LossNode>();
    compileLossNodes(generator, lossNodes, graph);
  }

  generator.generateFunctionFooter();
}
void GraphCompiler::compileBackward(Graph& graph,
                                    AbstractGenerator& generator) {
  auto graphTraversal = graph.traverse();
  generator.generateFunctionHeader("optimizeGraph");
  compileDerivatives(generator, graphTraversal, *graph.getOptimizer(), graph);
  generator.generateFunctionFooter();
}
} // namespace athena::core
