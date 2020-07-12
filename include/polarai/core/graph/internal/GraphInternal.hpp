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

#pragma once

#include <polar_core_export.h>
#include <polarai/core/Entity.hpp>
#include <polarai/core/context/internal/ContextInternal.hpp>
#include <polarai/core/graph/EdgeMark.hpp>
#include <polarai/core/node/internal/InputNodeInternal.hpp>
#include <polarai/utils/Index.hpp>
#include <polarai/utils/internal/TupleContainers.hpp>
#include <polarai/utils/string/StringView.hpp>

#include <queue>
#include <unordered_map>
#include <vector>

namespace polarai::core::internal {
using Topology = std::vector<Edge>;

class POLAR_CORE_EXPORT GraphInternal : public Entity {
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

  std::tuple<utils::Index, utils::Index>
  getGradient(utils::Index targetNodeIndex);

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
  createGradientGraph(utils::Index targetNodeIndex) const;

  utils::Index createInitialGradientNode(GraphInternal& gradientGraph,
                                         const NodeState* nodeStatePtr) const;

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
inline void GraphInternal::addToGraph(utils::Index index) {}

template <>
inline void GraphInternal::addToGraph<InputNodeInternal>(utils::Index index) {
  mInputNodeIndexes.emplace_back(index);
}

} // namespace polarai::core::internal
