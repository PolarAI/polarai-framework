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

#include <athena/core/InputNode.h>
#include <athena/core/LossNode.h>
#include <athena/core/Node.h>
#include <athena/core/OutputNode.h>
#include <athena/model/DotModel.h>

using namespace athena::core;

namespace athena::model {

enum TensorType { INNER, DERIVATIVE, ERR };

static void processTensor(core::inner::Tensor& tensor,
                          TensorType type,
                          std::ostream& stream) {
    stream << "\"" << &tensor << "\"[label=\"" << &tensor << "\\n";
    if (type == INNER) {
        stream << "INNER\\n";
    } else if (type == DERIVATIVE) {
        stream << "DERIVATIVE\\n";
    } else if (type == ERR) {
        stream << "ERR\\n";
    }
    stream << "vaddr: " << tensor.getVirtualAddress();
    auto shape = tensor.getShapeView();
    stream << "\\nShape: <";
    for (size_t idx : shape) {
        stream << idx << ", ";
    }
    stream << ">";
    stream << "\"]\n";
}

static void processInputNode(core::InputNode& inputNode, std::ostream& stream) {
    stream << "subgraph cluster_" << inputNode.getName() << " {";
    stream << "label=\"Input\";\n";
    stream << "style=filled;\n";
    if (inputNode.isFrozen()) {
        stream << "color=\"#91D2FF\";\n";
    } else {
        stream << "color=\"#FFDF9E\";\n";
    }
    stream << inputNode.getName() << "[label=\"" << inputNode.getName()
           << "\"];\n";
    auto& tensor = inner::getTensorFromNode(inputNode);
    processTensor(tensor, INNER, stream);
    stream << "}\n";
}

static void processActionNode(
    core::Context* mContext,
    core::Node& actionNode,
    const inner::NodeDependencies<core::Node>& nodeDeps,
    std::ostream& stream) {
    stream << "subgraph cluster_" << actionNode.getName() << " {";
    stream << "label=\"Action\";\n";
    stream << actionNode.getName() << " [label=\"" << actionNode.getName()
           << "\"];\n";
    auto& tensor = inner::getTensorFromNode(actionNode);
    processTensor(tensor, INNER, stream);
    for (size_t i = 0; i < actionNode.getOperation().getOperandsCount(); i++) {
        auto& dTensor = inner::getDerivativeTensor(actionNode, i);
        processTensor(dTensor, DERIVATIVE, stream);
    }
    auto& eTensor = inner::getIncomingGradient(actionNode);
    processTensor(eTensor, ERR, stream);
    stream << "}\n";

    for (auto& parent : nodeDeps.input) {
        auto* pNode = core::inner::getNodeTable(*mContext)[parent.nodeIndex];
        stream << pNode->getName() << " -> " << actionNode.getName();
        stream << ";\n";
    }
}

static void processLossNode(
    core::Context* mContext,
    core::LossNode& lossNode,
    const inner::NodeDependencies<core::LossNode>& nodeDeps,
    std::ostream& stream) {
    stream << "subgraph cluster_" << lossNode.getName() << " {";
    stream << "label=\"Loss\";\n";
    stream << lossNode.getName() << " [label=\"" << lossNode.getName()
           << "\"];\n";
    auto& tensor = inner::getTensorFromNode(lossNode);
    processTensor(tensor, INNER, stream);
    for (size_t i = 0; i < lossNode.getOperation().getOperandsCount(); i++) {
        auto& dTensor = inner::getDerivativeTensor(lossNode, i);
        processTensor(dTensor, DERIVATIVE, stream);
    }
    auto& eTensor = inner::getIncomingGradient(lossNode);
    stream << "}\n";

    for (auto& parent : nodeDeps.input) {
        auto* pNode = core::inner::getNodeTable(*mContext)[parent.nodeIndex];
        stream << pNode->getName() << " -> " << lossNode.getName();
        stream << ";\n";
    }
}

static void processOutputNode(
    core::Context* mContext,
    core::OutputNode& outputNode,
    const inner::NodeDependencies<core::OutputNode>& nodeDeps,
    std::ostream& stream) {
    stream << "subgraph cluster_" << outputNode.getName() << " {";
    stream << "label=\"Output\";\n";
    stream << outputNode.getName() << " [label=\"" << outputNode.getName()
           << "\"];\n";
    auto& tensor = inner::getTensorFromNode(outputNode);
    processTensor(tensor, INNER, stream);
    stream << "}\n";

    for (auto& parent : nodeDeps.input) {
        auto* pNode = core::inner::getNodeTable(*mContext)[parent.nodeIndex];
        stream << pNode->getName() << " -> " << outputNode.getName();
        stream << ";\n";
    }
}

void DotModel::exportGraph(Graph& graph, std::ostream& stream) {
    stream << "digraph " << graph.getGraphName() << " {\n";
    auto mTraversal = graph.traverse();
    auto* mContext = &inner::getContext(graph);
    for (auto& cluster : mTraversal.getClusters()) {
        auto& inputNodes = cluster.get<core::InputNode>();
        for (auto& nodeDeps : inputNodes) {
            auto& inputNode = node_cast<core::InputNode&>(
                *core::inner::getNodeTable(*mContext)[nodeDeps.nodeIndex]);
            processInputNode(inputNode, stream);
        }

        auto& actionNodes = cluster.get<core::Node>();
        for (auto& nodeDeps : actionNodes) {
            auto& actionNode = node_cast<core::Node&>(
                *core::inner::getNodeTable(*mContext)[nodeDeps.nodeIndex]);
            processActionNode(mContext, actionNode, nodeDeps, stream);
        }

        auto& lossNodes = cluster.get<core::LossNode>();
        for (auto& nodeDeps : lossNodes) {
            auto& lossNode = node_cast<core::LossNode&>(
                *core::inner::getNodeTable(*mContext)[nodeDeps.nodeIndex]);
            processLossNode(mContext, lossNode, nodeDeps, stream);
        }

        auto& outputNodes = cluster.get<core::OutputNode>();
        for (auto& nodeDeps : outputNodes) {
            auto& outputNode = node_cast<core::OutputNode&>(
                *core::inner::getNodeTable(*mContext)[nodeDeps.nodeIndex]);
            processOutputNode(mContext, outputNode, nodeDeps, stream);
        }
    }
    stream << "}\n";
}
}  // namespace athena::model