//===----------------------------------------------------------------------===//
// Copyright (c) 2020 Athena. All rights reserved.
// https://getathena.ml
//
// Licensed under MIT license.
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS, WITHOUT
// WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the
// License for the specific language governing permissions and limitations under
// the License.
//===----------------------------------------------------------------------===//

#include <athena/core/graph/internal/GraphCompiler.h>
#include <athena/core/node/internal/AbstractNodeInternal.h>
#include <athena/core/node/internal/NodeInternal.h>
#include <athena/utils/STLExtras.h>

namespace athena::core::internal {
void GraphCompiler::compile(Graph& graph, Generator& generator) {
  auto traversal = graph.traverse();
  const auto& clusters = traversal.getClusters();

  auto genGraph = generator.createGraph(graph.getName().getString(),
                                        graph.getPublicIndex());

  std::map<utils::Index, GenValue> nodeResults;

  auto ctxInternal = graph.getContext().internal();
  for (auto c : utils::enumerate(clusters)) {
    for (const auto& node : c.value().content) {
      auto topLevelInsPoint = generator.getInsertionPoint();
      auto& nodeInternal =
          ctxInternal->get<AbstractNodeInternal>(node.nodeIndex);
      std::vector<TensorInternal*> inputs;
      for (auto inp : node.input) {
        auto& incNode = ctxInternal->get<AbstractNodeInternal>(inp.nodeIndex);
        // fixme do not use const_cast
        inputs.push_back(const_cast<TensorInternal*>(incNode.getTensorPtr()));
      }
      // fixme don't use const_cast
      auto genNode = generator.createNode(
          nodeInternal.getName().getString(), node.nodeIndex, c.index(), inputs,
          *const_cast<TensorInternal*>(nodeInternal.getTensorPtr()));

      generator.setInsertionPoint(genNode);

      generator.callBuiltin<builtin::Alloc>(genNode.getResult());

      if (nodeInternal.getType() == NodeType::INPUT) {
        auto inpNode = ctxInternal->get<InputNodeInternal>(node.nodeIndex);
        generator.callBuiltin<builtin::Lock>(genNode.getResult(),
                                             LockType::READ_WRITE);
        generator.callBuiltin<builtin::InvokeLoader>(genNode.getResult());
        generator.callBuiltin<builtin::Return>(genNode.getResult());
      } else if (nodeInternal.getType() == NodeType::DEFAULT) {
        auto defaultNode = ctxInternal->get<NodeInternal>(node.nodeIndex);
        generator.callBuiltin<builtin::Lock>(genNode.getResult(),
                                             LockType::READ_WRITE);
        std::vector<utils::Index> idx;
        for (auto inp : node.input) {
          idx.push_back(inp.nodeIndex);
        }
        // fixme this must accept GenNode and return GenValue.
        auto opResult = defaultNode.getOperationPtr()->gen(
            ctxInternal, generator, idx, genNode);

        generator.callBuiltin<builtin::Return>(opResult);
      }

      // todo OutputNodes are not generated. Should they?

      generator.setInsertionPoint(genGraph);

      std::vector<GenValue> nodes;
      for (auto idx : node.input) {
        nodes.push_back(nodeResults.at(idx.nodeIndex));
      }
      auto res =
          generator.callBuiltin<builtin::NodeEval>(genGraph, genNode, nodes);
      nodeResults[node.nodeIndex] = res;
      generator.setInsertionPoint(topLevelInsPoint);
    }

    auto checkoint = generator.getInsertionPoint();
    generator.setInsertionPoint(genGraph);
    generator.callBuiltin<builtin::Barrier>(c.index());
    generator.setInsertionPoint(checkoint);
  }
}
} // namespace athena::core::internal
