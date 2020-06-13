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

#ifndef ATHENA_TRAVERSAL_H
#define ATHENA_TRAVERSAL_H

#include <athena/core/core_export.h>
#include <athena/utils/internal/TupleContainers.h>
#include <athena/utils/error/FatalError.h>

#include <set>
#include <vector>

namespace athena::core {
struct ATH_CORE_EXPORT Dependency {
  size_t nodeIndex;
  size_t nodeStateIndex;
  size_t clusterIndex;
  size_t mark;
  Dependency(size_t nodeIndex, size_t mark)
      : nodeIndex(nodeIndex), nodeStateIndex{}, clusterIndex{}, mark(mark) {}
};

struct ATH_CORE_EXPORT NodeState {
  explicit NodeState(bool isWayToFrozen)
      : nodeIndex{}, inputsCount{},
        isWayToFrozen{isWayToFrozen}, input{}, output{} {}
  NodeState()
      : nodeIndex{}, inputsCount{}, isWayToFrozen{true}, input{}, output{} {}
  NodeState(size_t nodeIndex, size_t inputsCount, bool isWayToFrozen,
            std::vector<Dependency> input, std::vector<Dependency> output)
      : nodeIndex(nodeIndex), inputsCount(inputsCount),
        isWayToFrozen(isWayToFrozen), input(std::move(input)),
        output(std::move(output)) {}

  static const Dependency& findDependency(const std::vector<Dependency>& dependence, int64_t mark) {
    for (auto& dep : dependence) {
      if (dep.mark == mark) {
        return dep;
      }
    }
    utils::FatalError(utils::ATH_BAD_ACCESS, "Access by incorrect mark.");
    return dependence[0];
  }

  size_t nodeIndex;
  size_t inputsCount;
  bool isWayToFrozen;
  std::vector<Dependency> input;
  std::vector<Dependency> output;
};

struct ATH_CORE_EXPORT Cluster {
  size_t nodeCount;
  std::vector<NodeState> content;
};

using Clusters = std::vector<Cluster>;

/**
 * Graph traversal
 */
class ATH_CORE_EXPORT Traversal {
private:
  Clusters mClusters;
  bool mIsValidTraversal;

public:
  [[nodiscard]] Clusters& clusters() { return mClusters; }
  [[nodiscard]] const Clusters& getClusters() const { return mClusters; }
  [[nodiscard]] bool isValidTraversal() const { return mIsValidTraversal; }
  bool& validTraversalFlag() { return mIsValidTraversal; }
};
} // namespace athena::core

#endif // ATHENA_TRAVERSAL_H
