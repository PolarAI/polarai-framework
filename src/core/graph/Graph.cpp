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

#include <athena/core/Graph.h>
#include <athena/core/InputNode.h>
#include <athena/core/LossNode.h>
#include <athena/core/Node.h>
#include <athena/core/OutputNode.h>
#include <athena/core/Traversal.h>
#include <athena/core/inner/InnerFunctions.h>

#include <algorithm>
#include <queue>

namespace athena::core {
template <typename TemplateNodeType>
void initVisitsOf(const OwningStorage& storage,
                  std::unordered_map<size_t, inner::NodeState>& visits) {
    for (auto& node : std::get<std::vector<TemplateNodeType>>(storage)) {
        visits[node.getNodeIndex()] = inner::NodeState{0};
    }
}
void initVisitsOf(const SyncStorage& storage,
                  std::unordered_map<size_t, inner::NodeState>& visits) {
    for (auto& index : storage) {
        visits[index] = inner::NodeState{0};
    }
}
void initVisits(std::unordered_map<size_t, inner::NodeState>& visits,
                const OwningStorage& owningStorage,
                const SyncStorage& syncStorage) {
    initVisitsOf(syncStorage, visits);
    initVisitsOf<InputNode>(owningStorage, visits);
    initVisitsOf<Node>(owningStorage, visits);
    initVisitsOf<OutputNode>(owningStorage, visits);
    initVisitsOf<LossNode>(owningStorage, visits);
}
void initQueue(std::queue<size_t>& queue,
               const OwningStorage& owningStorage,
               const SyncStorage& syncStorage,
               Context& context) {
    for (auto& inputNode : std::get<std::vector<InputNode>>(owningStorage)) {
        queue.push(inputNode.getNodeIndex());
    }
    for (auto& nodeIndex : syncStorage) {
        if (inner::getNodeTable(context)[nodeIndex]->getType() ==
            NodeType::INPUT) {
            queue.push(nodeIndex);
        }
    }
}
Graph::Graph(Context& context)
    : mContext(&context),
      mGraphIndex(inner::getGraphTable(*mContext).registerRecord(this)),
      mGraphName("MainGraph") {}
Graph::Graph(Graph&& rhs) noexcept
    : mSyncStorage(std::move(rhs.mSyncStorage)),
      mOwningStorage(std::move(rhs.mOwningStorage)),
      mTopology(std::move(rhs.mTopology)),
      mContext(rhs.mContext),
      mGraphIndex(rhs.mGraphIndex),
      mTraversal(std::move(rhs.mTraversal)) {
    inner::getGraphTable(*mContext)[mGraphIndex] = this;
    rhs.fullClear();
}
Graph::~Graph() {
    inner::getGraphTable(*mContext)[mGraphIndex] = nullptr;
    for (auto indexNode : mSyncStorage) {
        if (auto* node = inner::getNodeTable(*mContext)[indexNode]; node) {
            inner::setGraphIndex(*(node), inner::kKUndefinedIndex);
        }
    }
}
template <typename TemplateNodeType>
void Graph::saveRealNode(TemplateNodeType& node,
                         bool isRepairedNode,
                         bool isErase) {
    if (isErase) {
        mSyncStorage.erase(node.getNodeIndex());
    }
    std::get<std::vector<TemplateNodeType>>(mOwningStorage)
        .emplace_back(std::move(node));
}
void Graph::fullClear() {
    clear();
    mGraphIndex = inner::kKUndefinedIndex;
}
const SyncStorage& Graph::getSyncStorage() const {
    return mSyncStorage;
}
const OwningStorage& Graph::getOwningStorage() const {
    return mOwningStorage;
}
const Topology& Graph::getTopology() const {
    return mTopology;
}
void Graph::addNode(AbstractNode& node) {
    if (Graph* graphPointer =
            inner::getGraphTable(*mContext)[node.getGraphIndex()];
        graphPointer) {
        FatalError(ATH_FATAL_OTHER, "addNode() in Graph : ", this,
                   ". GraphIndex : ", mGraphIndex,
                   ". Adding Node to the graph to which it does not belong");
    }
    mSyncStorage.insert(node.getNodeIndex());
    inner::setGraphIndex(node, mGraphIndex);
    inner::setTraversalValidity(mTraversal, false);
}
void Graph::saveNode(AbstractNode& node, bool isRepairedNode, bool isErase) {
    if (node.getGraphIndex() != mGraphIndex) {
        FatalError(ATH_FATAL_OTHER, "saveNode() in Graph : ", this,
                   ". GraphIndex : ", mGraphIndex,
                   ". Saving Node in the graph to which it does not belong");
    }
    switch (node.getType()) {
        case NodeType::DEFAULT:
            saveRealNode(node_cast<Node&>(node), isRepairedNode, isErase);
            break;
        case NodeType::INPUT:
            saveRealNode(node_cast<InputNode&>(node), isRepairedNode, isErase);
            break;
        case NodeType::OUTPUT:
            saveRealNode(node_cast<OutputNode&>(node), isRepairedNode, isErase);
            break;
        case NodeType::LOSS:
            saveRealNode(node_cast<LossNode&>(node), isRepairedNode, isErase);
            break;
        default:
            FatalError(ATH_FATAL_OTHER, "saveNode() in Graph : ", this,
                       ". GraphIndex : ", mGraphIndex, ". Undefined node type");
    }
}
void Graph::saveNode(AbstractNode& node, bool isRepairedNode) {
    saveNode(node, isRepairedNode, true);
}
void Graph::saveAllSyncNodes(bool isRepairedNode) {
    for (auto syncNode : mSyncStorage) {
        saveNode(*inner::getNodeTable(*mContext)[syncNode], isRepairedNode,
                 false);
    }
    mSyncStorage.clear();
}
void Graph::removeNode(AbstractNode& node) {
    mSyncStorage.erase(node.getNodeIndex());
    size_t nodeIndex = node.getNodeIndex();
    auto removePredicate = [nodeIndex](const inner::Edge& edge) -> bool {
        return nodeIndex == edge.startNodeIndex ||
               nodeIndex == edge.endNodeIndex;
    };
    mTopology.erase(
        std::remove_if(mTopology.begin(), mTopology.end(), removePredicate),
        mTopology.end());
    inner::setTraversalValidity(mTraversal, false);
}
void Graph::link(const AbstractNode& startNode,
                 const AbstractNode& endNode,
                 EdgeMark mark) {
    if (endNode.getType() == NodeType::INPUT) {
        FatalError(ATH_FATAL_OTHER, "link() in Graph : ", this,
                   ". GraphIndex : ", mGraphIndex,
                   ". End node of edge can not be InputNode");
    }
    if (startNode.getGraphIndex() == endNode.getGraphIndex() &&
        startNode.getGraphIndex() == mGraphIndex) {
        mTopology.emplace_back(startNode.getNodeIndex(), endNode.getNodeIndex(),
                               mark);
        inner::incrementInputCount(
            *inner::getNodeTable(*mContext)[endNode.getNodeIndex()]);
        inner::setTraversalValidity(mTraversal, false);
    } else {
        FatalError(ATH_FATAL_OTHER, "link() in Graph : ", this,
                   ". GraphIndex : ", mGraphIndex,
                   ". Nodes belong to different graphs");
    }
}
void Graph::clear() {
    mSyncStorage.clear();
    mTopology.clear();
    OwningStorage emptyStorage;
    mOwningStorage.swap(emptyStorage);
    inner::getClusters(mTraversal).clear();
    inner::setTraversalValidity(mTraversal, false);
}
size_t Graph::countOwningNodes() const {
    return std::get<std::vector<Node>>(mOwningStorage).size() +
           std::get<std::vector<InputNode>>(mOwningStorage).size();
}
size_t Graph::countSyncNodes() const {
    return mSyncStorage.size();
}
size_t Graph::getGraphIndex() const {
    return mGraphIndex;
}
bool Graph::isValidTraversal() const {
    return mTraversal.isValidTraversal();
}

#define TRAVERSE_ADD_NODES_TO_CLUSTER(typeName)                       \
    case typeName:                                                    \
        cluster.get<NodeTypeId<typeName>::type>().emplace_back(       \
            node->getNodeIndex(), std::move(visits[nodeIndex].input), \
            std::move(visits[nodeIndex].output));                     \
        break;

const Traversal& Graph::traverse() {
    if (mTraversal.isValidTraversal()) {
        return mTraversal;
    }
    inner::getClusters(mTraversal).clear();
    std::sort(mTopology.begin(), mTopology.end());
    std::queue<size_t> currentQueue, newQueue;
    std::unordered_map<size_t, inner::NodeState> visits;
    initVisits(visits, mOwningStorage, mSyncStorage);
    initQueue(currentQueue, mOwningStorage, mSyncStorage, *mContext);
    while (true) {
        inner::Cluster cluster{0};
        while (!currentQueue.empty()) {
            size_t nodeIndex = currentQueue.front();
            currentQueue.pop();
            inner::Edge target(nodeIndex, 0, 0);
            auto edgeIterator =
                std::lower_bound(mTopology.begin(), mTopology.end(), target);
            while (edgeIterator != mTopology.end() &&
                   edgeIterator->startNodeIndex == nodeIndex) {
                auto& inputCount =
                    visits[edgeIterator->endNodeIndex].inputCount;
                ++inputCount;
                auto targetInputCount =
                    inner::getNodeTable(*mContext)[edgeIterator->endNodeIndex]
                        ->getInputsCount();
                if (inputCount == targetInputCount) {
                    newQueue.push(edgeIterator->endNodeIndex);
                } else if (inputCount > targetInputCount) {
                    FatalError(ATH_FATAL_OTHER,
                               "traverse() in Graph: ", mGraphIndex,
                               ". Graph has cycle(s)");
                }
                visits[edgeIterator->endNodeIndex].input.emplace(
                    nodeIndex, edgeIterator->mark);
                visits[nodeIndex].output.emplace(edgeIterator->endNodeIndex,
                                                 edgeIterator->mark);
                ++edgeIterator;
            }
            AbstractNode* node = inner::getNodeTable(*mContext)[nodeIndex];
            switch (node->getType()) {
                TRAVERSE_ADD_NODES_TO_CLUSTER(NodeType::DEFAULT)
                TRAVERSE_ADD_NODES_TO_CLUSTER(NodeType::INPUT)
                TRAVERSE_ADD_NODES_TO_CLUSTER(NodeType::LOSS)
                TRAVERSE_ADD_NODES_TO_CLUSTER(NodeType::OUTPUT)
                default:
                    FatalError(ATH_NOT_IMPLEMENTED,
                               "Undefined NodeType in traverse()");
            }
            ++cluster.nodeCount;
        }
        if (cluster.nodeCount > 0) {
            inner::getClusters(mTraversal).emplace_back(std::move(cluster));
        }
        std::swap(currentQueue, newQueue);
        if (currentQueue.empty()) {
            break;
        }
    }

    // Now that we have graph traversal, it is possible to determine tensor
    // shapes
    setUpTensors();

    inner::setTraversalValidity(mTraversal, true);
    return mTraversal;
}

#undef TRAVERSE_ADD_NODES_TO_CLUSTER

void Graph::setUpTensors() const {
    for (auto& cluster : mTraversal.getClusters()) {
        auto& actionNodes = cluster.get<Node>();
        for (auto& nodeDep : actionNodes) {
            std::vector<inner::Tensor*> opArgs;

            auto& node = node_cast<Node&>(
                *inner::getNodeTable(*mContext)[nodeDep.nodeIndex]);
            std::for_each(
                nodeDep.input.begin(), nodeDep.input.end(),
                [&](const auto& inp) {
                    opArgs.push_back(&inner::getTensorFromNode(
                        *inner::getNodeTable(*mContext)[inp.nodeIndex]));
                });

            inner::setResultTensor(
                node, node.getOperation().getResultTensor(*mContext, opArgs));
            inner::setErrorTensor(
                node, node.getOperation().getErrorTensor(
                          *mContext, opArgs, mOptimizer->getRequiredOrder()));

            for (size_t idx = 0; idx < node.getOperation().getOperandsCount();
                 idx++) {
                auto& derivativeTensor =
                    *node.getOperation().getDerivativeTensor(*mContext, opArgs,
                                                             idx);
                inner::addDerivativeTensor(node, derivativeTensor);
            }
        }

        auto& lossNodes = cluster.get<LossNode>();
        for (auto& nodeDep : lossNodes) {
            std::vector<inner::Tensor*> opArgs;

            auto& node = node_cast<LossNode&>(
                *inner::getNodeTable(*mContext)[nodeDep.nodeIndex]);
            std::for_each(
                nodeDep.input.begin(), nodeDep.input.end(),
                [&](const auto& inp) {
                    opArgs.push_back(&inner::getTensorFromNode(
                        *inner::getNodeTable(*mContext)[inp.nodeIndex]));
                });

            inner::setResultTensor(
                node, node.getOperation().getResultTensor(*mContext, opArgs));

            for (size_t idx = 0; idx < node.getOperation().getOperandsCount();
                 idx++) {
                auto& derivativeTensor =
                    *node.getOperation().getDerivativeTensor(*mContext, opArgs,
                                                             idx);
                // For loss node error and derivative means the same
                inner::addDerivativeTensor(node, derivativeTensor);
            }
        }

        auto& outputNodes = cluster.get<OutputNode>();
        for (auto& nodeDep : outputNodes) {
            auto& node = node_cast<OutputNode&>(
                *inner::getNodeTable(*mContext)[nodeDep.nodeIndex]);
            auto& parentNode = *inner::getNodeTable(
                *mContext)[nodeDep.input.begin()->nodeIndex];
            inner::setResultTensor(node, &inner::getTensorFromNode(parentNode));
        }
    }
}
}  // namespace athena::core

namespace athena::core::inner {
Context& getContext(athena::core::Graph& graph) {
    return *(graph.mContext);
}
}  // namespace athena::core::inner