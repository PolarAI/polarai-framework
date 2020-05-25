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

#ifndef ATHENA_GRAPHINTERNAL_H
#define ATHENA_GRAPHINTERNAL_H

#include <athena/core/core_export.h>

#include <athena/core/Entity.h>
#include <athena/core/context/internal/ContextInternal.h>
#include <athena/core/graph/EdgeMark.h>
#include <athena/utils/Index.h>
#include <athena/utils/internal/TupleContainers.h>
#include <athena/utils/string/StringView.h>

#include <athena/core/node/internal/InputNodeInternal.h>
#include <queue>
#include <unordered_map>

namespace athena::core::internal {
using Topology = std::vector<Edge>;

class ATH_CORE_EXPORT GraphInternal : public Entity {
public:
  explicit GraphInternal(utils::WeakPtr<ContextInternal> context,
                         utils::Index publicGraphIndex,
                         utils::String name = utils::String(""));

  ~GraphInternal() override = default;

  GraphInternal(GraphInternal&&) = default;

  template <typename TemplateNodeTypeInternal>
  void addToGraph(utils::Index index);

  template <typename TemplateNodeTypeInternal, typename... Args>
  utils::Index create(Args&&... args) {
    auto index = mContext.lock()->create<TemplateNodeTypeInternal>(
        mContext.lock(), mContext.lock()->getNextPublicIndex(),
        std::forward<Args>(args)...);
    // TODO try to use "enable shared from this" for deleting  "mContext,
    // mContext.lock()->getNextPublicIndex()"
    addToGraph<TemplateNodeTypeInternal>(index);
    return index;
  }

  void connect(utils::Index startNode, utils::Index endNode, EdgeMark edgeMark);

  const Traversal& traverse();

  void setUpTensors() const;

  std::tuple<utils::Index, utils::Index> getGradient();

private:
  struct NodeStateIndex {
    size_t clusterIndex{};
    size_t nodeStateIndex{};
  };

  void bypassDependenceOfCurrentNodeState(
      const NodeState& currentNodeState, size_t currentClusterIndex,
      size_t currentNodeStateIndex,
      std::unordered_map<utils::Index, NodeState>& nodeStates,
      std::unordered_map<utils::Index, NodeStateIndex>& traversedNodeStates);

  void initInputNodeStates(std::unordered_map<utils::Index, NodeState>&
                               isPartOfWayToUnfrozenFlags) const;

  std::tuple<utils::Index, std::unordered_map<utils::Index, utils::Index>>
  createGradientGraph() const;

  utils::Index createInitialGradientNode(const NodeState* nodeStatePtr) const;

  utils::Index accumulateOutputNodes(
      GraphInternal& gradient, const NodeState* nodeStatePtr,
      const std::unordered_map<const NodeState*, utils::Index>&
          mapNodeStateToFinalGradientIndex) const;

  void mergeEdges(const std::vector<core::internal::Edge>& edges);

  utils::Index createWeightChangingGraph(
      const std::unordered_map<utils::Index, utils::Index>& mapInputNodes);

  Topology mTopology;
  Traversal mTraversal;
  std::vector<utils::Index> mInputNodeIndexes;
  std::unordered_map<utils::Index, size_t> mInputsCount;
  std::unordered_map<utils::Index, size_t> mOutputsCount;
};

template <typename TemplateNodeTypeInternal>
void GraphInternal::addToGraph(utils::Index index) {}

} // namespace athena::core::internal

#endif // ATHENA_GRAPHINTERNAL_H
