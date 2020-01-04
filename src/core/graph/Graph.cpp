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

using namespace athena::core::inner;

namespace athena::core {
//template <typename TemplateNodeType>
//void initVisitsOf(const OwningStorage& storage,
//                  std::unordered_map<size_t, NodeState>& visits) {
//    for (auto& node : std::get<std::vector<TemplateNodeType>>(storage)) {
//        visits[node.getNodeIndex()] = NodeState{0};
//    }
//}
//void initVisitsOf(const SyncStorage& storage,
//                  std::unordered_map<size_t, NodeState>& visits) {
//    for (auto& index : storage) {
//        visits[index] = NodeState{0};
//    }
//}
//void initVisits(std::unordered_map<size_t, NodeState>& visits,
//                const OwningStorage& owningStorage,
//                const SyncStorage& syncStorage) {
//    initVisitsOf(syncStorage, visits);
//    initVisitsOf<InputNode>(owningStorage, visits);
//    initVisitsOf<Node>(owningStorage, visits);
//    initVisitsOf<OutputNode>(owningStorage, visits);
//    initVisitsOf<LossNode>(owningStorage, visits);
//}
void initQueue(std::queue<size_t>& queue,
               const OwningStorage& owningStorage,
               const SyncStorage& syncStorage,
               Context& context) {
    for (auto& inputNode : std::get<std::vector<InputNode>>(owningStorage)) {
        queue.push(inputNode.getNodeIndex());
    }
    for (auto& nodeIndex : syncStorage) {
        if (getNodeTable(context)[nodeIndex]->getType() == NodeType::INPUT) {
            queue.push(nodeIndex);
        }
    }
}
Graph::Graph(Context& context)
    : mContext(&context),
      mGraphIndex(getGraphTable(*mContext).registerRecord(this)),
      mGraphName("MainGraph") {}
Graph::Graph(Graph&& rhs) noexcept
    : mSyncStorage(std::move(rhs.mSyncStorage)),
      mOwningStorage(std::move(rhs.mOwningStorage)),
      mTopology(std::move(rhs.mTopology)),
      mContext(rhs.mContext),
      mGraphIndex(rhs.mGraphIndex),
      mTraversal(std::move(rhs.mTraversal)) {
    getGraphTable(*mContext)[mGraphIndex] = this;
    rhs.fullClear();
}
Graph::~Graph() {
    getGraphTable(*mContext)[mGraphIndex] = nullptr;
    for (auto indexNode : mSyncStorage) {
        if (auto* node = getNodeTable(*mContext)[indexNode]; node) {
            setGraphIndex(*(node), kKUndefinedIndex);
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
    std::get<std::vector<TemplateNodeType>>(mOwningStorage).emplace_back(std::move(node));
//    if (isRepairedNode) {
//        TemplateNodeType newNode(
//            std::get<std::vector<TemplateNodeType>>(mOwningStorage).back());
//        node = std::move(newNode);
//    }
}
void Graph::fullClear() {
    clear();
    mGraphIndex = kKUndefinedIndex;
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
    if (Graph* graphPointer = getGraphTable(*mContext)[node.getGraphIndex()];
        graphPointer) {
        FatalError(ATH_FATAL_OTHER, "addNode() in Graph : ", this,
                   ". GraphIndex : ", mGraphIndex,
                   ". Adding Node to the graph to which it does not belong");
    }
    mSyncStorage.insert(node.getNodeIndex());
    setGraphIndex(node, mGraphIndex);
    setTraversalValidity(mTraversal, false);
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
        saveNode(*getNodeTable(*mContext)[syncNode], isRepairedNode, false);
    }
    mSyncStorage.clear();
}
void Graph::removeNode(AbstractNode& node) {
    mSyncStorage.erase(node.getNodeIndex());
    size_t nodeIndex = node.getNodeIndex();
    auto removePredicate = [nodeIndex](const Edge& edge) -> bool {
        return nodeIndex == edge.startNodeIndex ||
               nodeIndex == edge.endNodeIndex;
    };
    mTopology.erase(
        std::remove_if(mTopology.begin(), mTopology.end(), removePredicate),
        mTopology.end());
    setTraversalValidity(mTraversal, false);
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
        incrementInputCount(
            *getNodeTable(*mContext)[endNode.getNodeIndex()]);
        setTraversalValidity(mTraversal, false);
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
    getClusters(mTraversal).clear();
    setTraversalValidity(mTraversal, false);
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
    getClusters(mTraversal).clear();
    std::sort(mTopology.begin(), mTopology.end());
    std::queue<size_t> currentQueue, newQueue;
    std::unordered_map<size_t, NodeState> visits;
    //initVisits(visits, mOwningStorage, mSyncStorage);
    initQueue(currentQueue, mOwningStorage, mSyncStorage, *mContext);
    while (true) {
        Cluster cluster{0};
        while (!currentQueue.empty()) {
            size_t nodeIndex = currentQueue.front();
            currentQueue.pop();
            Edge target(nodeIndex, 0, 0);
            auto edgeIterator =
                std::lower_bound(mTopology.begin(), mTopology.end(), target);
            while (edgeIterator != mTopology.end() &&
                   edgeIterator->startNodeIndex == nodeIndex) {
                visits[edgeIterator->endNodeIndex].input[edgeIterator->mark] = nodeIndex;
                visits[edgeIterator->startNodeIndex].output.emplace(nodeIndex);
                auto inputsCount =
                    visits[edgeIterator->endNodeIndex].input.size();
                auto targetInputCount =
                    getNodeTable(*mContext)[edgeIterator->endNodeIndex]
                        ->getInputsCount();
                if (inputsCount == targetInputCount) {
                    newQueue.push(edgeIterator->endNodeIndex);
                } else if (inputsCount > targetInputCount) {
                    FatalError(ATH_FATAL_OTHER,
                               "traverse() in Graph: ", mGraphIndex,
                               ". Graph is have an cycle(s)");
                }
                ++edgeIterator;
            }
            AbstractNode* node = getNodeTable(*mContext)[nodeIndex];
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
            getClusters(mTraversal).emplace_back(std::move(cluster));
        }
        std::swap(currentQueue, newQueue);
        if (currentQueue.empty()) {
            break;
        }
    }

    // Now that we have graph traversal, it is possible to determine tensor
    // shapes
    setUpTensors();

    setTraversalValidity(mTraversal, true);
    return mTraversal;
}

#undef TRAVERSE_ADD_NODES_TO_CLUSTER

template <typename TemplateNodeType>
void createTensorsForNodeType(Context* context, const std::vector<NodeDependencies<TemplateNodeType>>& nodes) {
    for (auto& nodeDep : nodes) {
        auto operationArgs = getOperationArgs(*context, nodeDep);
        auto& node =
            node_cast<TemplateNodeType&>(*getNodeTable(*context)[nodeDep.nodeIndex]);
        setResultTensor(node, node.getOperation().createTensor(*context, operationArgs));
        for (auto& output : nodeDep.output) {
            addOutgoingDerivative(node, node.getOperation().createTensor(
                                            *context, operationArgs), output);
        }
    }
}

void Graph::setUpTensors() const {
    for (auto& cluster : mTraversal.getClusters()) {
        auto& actionNodes = cluster.get<Node>();
        createTensorsForNodeType(mContext, actionNodes);
        auto& lossNodes = cluster.get<LossNode>();
        createTensorsForNodeType(mContext, lossNodes);
        auto& outputNodes = cluster.get<OutputNode>();
        for (auto& nodeDep : outputNodes) {
            auto& node = node_cast<OutputNode&>(
                *getNodeTable(*mContext)[nodeDep.nodeIndex]);
            auto& parentNode = *inner::getNodeTable(
                *mContext)[nodeDep.input.begin()->second];
            setResultTensor(node, inner::getTensorSmartPtrFromNode(parentNode));
        }
    }
}
}  // namespace athena::core

namespace athena::core::inner {
Context& getContext(athena::core::Graph& graph) {
    return *(graph.mContext);
}
}