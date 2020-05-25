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

#include <athena/core/node/internal/AbstractNodeInternal.h>
#include <athena/core/node/internal/NodeInternal.h>
#include <athena/core/tensor/internal/TensorInternal.h>
#include <athena/model/DotModel.h>

using namespace athena::core;

namespace athena::model {
namespace {
static void processTensor(const internal::TensorInternal* tensor,
                          std::ostream& stream) {
  stream << "\"" << tensor << "\" [label=\""
         << "ResultTensorInfo\\n";
  stream << "Name: " << tensor->getName().getString();
  stream << "\\nAddress: " << tensor;
  stream << "\\nIndex: " << tensor->getPublicIndex();
  stream << "\\nVirtualAddress: " << tensor->getVirtualAddress();
  auto shape = tensor->getShapeView();
  stream << "\\nShape: <";
  for (size_t idx : shape) {
    stream << idx << ", ";
  }
  stream << ">";
  stream << "\"]\n";
}

const char* getStringNodeType(NodeType nodeType) {
  switch (nodeType) {
  case core::NodeType::DEFAULT:
    return "Action";
  case core::NodeType::OUTPUT:
    return "Output";
  case core::NodeType::INPUT:
    return "Input";
  default:
    return "Undefined";
  }
}

void processOperation(const internal::OperationInternal* operation,
                      std::ostream& stream) {
  stream << "\"" << operation << "\" [label=\""
         << "OperationInfo"
         << "\\n";
  stream << "Name=" << operation->getName().getString()
         << "\\nAddress: " << operation
         << "\\nOperationIndex: " << operation->getPublicIndex()
         << "\\nOperationName: " << operation->getName().getString();
  stream << "\"]\n";
}

void processNode(const Context& context, const NodeState& nodeState,
                 std::ostream& stream) {
  auto& node = context.internal()->getRef<internal::AbstractNodeInternal>(
      nodeState.nodeIndex);
  auto nodeType = node.getType();
  stream << "subgraph cluster_" << node.getName().getString() << " {";
  auto label = getStringNodeType(nodeType);
  stream << "label=\"" << label << "\";\n";
  stream << node.getName().getString() << " [label=\""
         << "NodeInfo"
         << "\\nName=" << node.getName().getString()
         << "\\nNodeId=" << node.getPublicIndex();
  stream << "\\nisWayToFrozen=" << std::boolalpha << nodeState.isWayToFrozen;
  stream << "\\n\\nOutputDependence:";
  for (size_t indexOutputDependence = 0;
       indexOutputDependence < nodeState.output.size();
       ++indexOutputDependence) {
    stream << "\\n" + std::to_string(indexOutputDependence + 1) +
                  ") NodeIndex: "
           << nodeState.output[indexOutputDependence].nodeIndex
           << ", Mark: " << nodeState.output[indexOutputDependence].mark;
  }

  stream << "\"];\n";
  auto tensor = context.internal()->getPtr<internal::TensorInternal>(
      node.getTensorIndex());
  processTensor(tensor, stream);
  if (nodeType == core::NodeType::DEFAULT) {
    auto& actionNode = static_cast<const internal::NodeInternal&>(node);
    processOperation(actionNode.getOperationPtr(), stream);
  }
  stream << "}\n";
  for (auto& input : nodeState.input) {
    auto& inputNode =
        context.internal()->getRef<internal::AbstractNodeInternal>(
            input.nodeIndex);
    stream << inputNode.getName().getString() << " -> "
           << node.getName().getString();
    stream << ";\n";
  }
}
} // namespace

void DotModel::exportGraph(Graph& graph, std::ostream& stream) {
  stream << "digraph " << graph.getName().getString() << " {\n";
  auto traversal = graph.traverse();
  auto context = graph.getContext();
  for (auto& cluster : traversal.getClusters()) {
    for (auto& nodeState : cluster.content) {
      processNode(context, nodeState, stream);
    }
  }
  stream << "}\n";
}
} // namespace athena::model
