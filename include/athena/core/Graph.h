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

#ifndef ATHENA_GRAPH_H
#define ATHENA_GRAPH_H

#include <athena/core/Traversal.h>
#include <athena/core/inner/Settings.h>
#include <athena/core/inner/Table.h>

#include <set>
#include <unordered_map>
#include <unordered_set>

namespace athena::core {
namespace inner {
struct Edge {
    size_t startNodeIndex;
    size_t endNodeIndex;
    EdgeMark mark;
    Edge(size_t startNodeIndex, size_t endNodeIndex, EdgeMark mark)
        : startNodeIndex(startNodeIndex),
          endNodeIndex(endNodeIndex),
          mark(mark) {}
    bool operator==(const Edge& rhs) const {
        return startNodeIndex == rhs.startNodeIndex &&
               endNodeIndex == rhs.endNodeIndex;
    }
    bool operator<(const Edge& rhs) const {
        return startNodeIndex < rhs.startNodeIndex;
    }
};
}  // namespace inner

using SyncStorage = std::unordered_set<size_t>;
using OwningStorage = inner::
    TupleContainers<std::vector, Node, InputNode, OutputNode, LossNode>::Holder;
using Topology = std::vector<inner::Edge>;

/**
 * A computation graph is an abstraction to represent an arbitrary function
 * in a way that is suitable for computation.
 */
class Graph {
    private:
    SyncStorage mSyncStorage;
    OwningStorage mOwningStorage;
    Topology mTopology;
    size_t mGraphIndex;
    Traversal mTraversal;
    template <typename TemplateNodeType>
    void saveRealNode(TemplateNodeType& node,
                      bool isRepairedNode,
                      bool isErase);
    void saveNode(AbstractNode& node, bool isRepairedNode, bool isErase);
    ATHENA_REINITIALIZE void fullClear();

    public:
    Graph();
    Graph(const Graph& rhs) = delete;
    Graph(Graph&& rhs) noexcept;
    ~Graph();

    Graph& operator=(const Graph& rhs) = delete;
    Graph& operator=(Graph&& rhs) noexcept;

    const SyncStorage& getSyncStorage() const;
    const OwningStorage& getOwningStorage() const;
    const Topology& getTopology() const;
    /**
     * Add node to Graph
     * @param node A node to be added
     */
    void addNode(AbstractNode& node);
    void saveNode(AbstractNode& node, bool isRepairedNode = true);
    void saveAllSyncNodes(bool isRepairedNode = true);
    /**
     * Remove node from Graph
     * @param node Node to be removed
     */
    void removeNode(AbstractNode& node);
    /**
     * Add oriented edge between two nodes
     * @param startNode Start Node
     * @param endNode End Node
     * @param mark
     */
    void link(const AbstractNode& startNode,
              const AbstractNode& endNode,
              EdgeMark mark);
    size_t countOwningNodes() const;
    size_t countSyncNodes() const;
    size_t getGraphIndex() const;
    /**
     * Resets object to initial state
     */
    void clear();
    /**
     *
     * @return if traversal is still valid
     */
    bool isValidTraversal() const;
    /**
     * Traverse current Graph and save the results inside object
     * @return A reference to result traversal
     */
    const Traversal& traverse();

    /**
     * Get last traversal for given Graph
     * @param graph An instance of Graph
     * @return A reference to last traversal
     */
    friend Traversal& inner::getTraversal(Graph& graph);
};
}  // namespace athena::core

#endif  // ATHENA_GRAPH_H
