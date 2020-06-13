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

#include <athena/core/graph/internal/GraphInternal.h>
#include <athena/core/node/internal/NodeInternal.h>
#include <athena/loaders/internal/ConstantLoaderInternal.h>
#include <athena/loaders/internal/DummyLoaderInternal.h>
#include <athena/operation/AddOperation.h>
#include <athena/operation/MulOperation.h>
#include <athena/operation/internal/AddOperationInternal.h>
#include <athena/operation/internal/MulOperationInternal.h>
#include <queue>

namespace athena::core::internal {
GraphInternal::GraphInternal(utils::WeakPtr<ContextInternal> context,
                             utils::Index publicGraphIndex, utils::String name)
    : Entity(std::move(context), publicGraphIndex, std::move(name)),
      mTopology{}, mTraversal{}, mInputNodeIndexes{}, mInputsCount{},
      mOutputsCount{} {}

template <typename Queue, typename Storage>
void initQueue(Queue& queue, const Storage& storage) {
  for (auto inputNodeIndex : storage) {
    queue.push(inputNodeIndex);
  }
}

template <typename Map, typename Getter>
size_t safetyIncrementMapValue(Map& map, utils::Index index, Getter& getter) {
  auto it = map.find(index);
  if (it != map.end()) {
    getter(map, index)++;
  } else {
    getter(map, index) = 1;
  }
  return getter(map, index);
}

void GraphInternal::connect(utils::Index startNode, utils::Index endNode,
                            EdgeMark edgeMark) {
  mTopology.emplace_back(startNode, endNode, edgeMark);
  auto getter = [](std::unordered_map<utils::Index, size_t>& map,
                   utils::Index index) -> size_t& { return map[index]; };
  safetyIncrementMapValue(mInputsCount, endNode, getter);
  safetyIncrementMapValue(mOutputsCount, startNode, getter);
}

std::tuple<std::vector<TensorInternal*>, std::unordered_map<int64_t, utils::Index>>
getOperationArgIndexes(utils::SharedPtr<ContextInternal>& context,
                       const NodeState& nodeState) {
  std::vector<TensorInternal*> operationArgs{};
  std::unordered_map<int64_t, utils::Index> mapMarkToLocalTensorIndex{};
  operationArgs.reserve(nodeState.input.size());
  size_t index = 0;
  for (auto& inputDependence : nodeState.input) {
    mapMarkToLocalTensorIndex[inputDependence.mark] = index;
    operationArgs.push_back(
        context->getPtr<AbstractNodeInternal>(inputDependence.nodeIndex)
            ->getTensorPtr());
    ++index;
  }
  return std::make_tuple(operationArgs, mapMarkToLocalTensorIndex);
}

void GraphInternal::setUpTensors() const {
  auto contextInternal = mContext.lock();
  for (auto& cluster : mTraversal.getClusters()) {
    for (auto& nodeState : cluster.content) {
      auto [args, mapMarkToLocalTensorIndex] = getOperationArgIndexes(contextInternal, nodeState);
      auto node =
          contextInternal->getPtr<AbstractNodeInternal>(nodeState.nodeIndex);
      if (node->getTensorIndex() != 0) {
        continue;
      }
      if (std::find(mInputNodeIndexes.begin(), mInputNodeIndexes.end(),
                    node->getPublicIndex()) != mInputNodeIndexes.end()) {
        continue;
      } else if (node->getType() == NodeType::DEFAULT) {
        auto funcNode = static_cast<NodeInternal*>(node);
        auto tensorIndex = funcNode->getOperationPtr()->createResultTensor(
            mContext.lock(), mapMarkToLocalTensorIndex, args);
        funcNode->setTensorIndex(tensorIndex);
      } else if (node->getType() == NodeType::OUTPUT) {
#ifdef DEBUG
        if (args.size() != 1) {
          utils::FatalError(utils::ATH_ASSERT,
                            "Error while setUpTensors() is working. Number "
                            "arguments coming to output node doesn't equal 1.");
        }
#endif
        node->setTensorIndex(args[0]->getPublicIndex());
      }
    }
  }
}

void GraphInternal::initInputNodeStates(
    std::unordered_map<utils::Index, NodeState>& isPartOfWayToUnfrozenFlags)
    const {
  auto context = mContext.lock();
  for (auto& indexInputNode : mInputNodeIndexes) {
    auto& abstractNode = context->getRef<AbstractNodeInternal>(indexInputNode);
    if (abstractNode.getType() == NodeType::INPUT) {
      auto& inputNode = static_cast<InputNodeInternal&>(abstractNode);
      isPartOfWayToUnfrozenFlags[indexInputNode] =
          NodeState{inputNode.isFrozen()};
    } else {
      isPartOfWayToUnfrozenFlags[indexInputNode] = NodeState{true};
    }
  }
}

void GraphInternal::bypassDependenceOfCurrentNodeState(
    const NodeState& currentNodeState, size_t currentClusterIndex,
    size_t currentNodeStateIndex,
    std::unordered_map<utils::Index, NodeState>& nodeStates,
    std::unordered_map<utils::Index, NodeStateIndex>& traversedNodeStates) {
  for (auto& dependence : currentNodeState.input) {
    auto nodeIndex = dependence.nodeIndex;
    std::vector<Dependency>* outputs{};
    if (nodeStates.find(nodeIndex) != nodeStates.end()) {
      outputs = &nodeStates.at(nodeIndex).output;
    } else {
      auto index = traversedNodeStates.at(nodeIndex);
      outputs = &mTraversal.clusters()[index.clusterIndex]
                     .content[index.nodeStateIndex]
                     .output;
    }
    for (auto& output : *outputs) {
      if (output.nodeIndex == currentNodeState.nodeIndex) {
        output.clusterIndex = currentClusterIndex;
        output.nodeStateIndex = currentNodeStateIndex;
      }
    }
  }
  for (auto& dependence : currentNodeState.output) {
    auto nodeIndex = dependence.nodeIndex;
    std::vector<Dependency>* inputs{};
    if (nodeStates.find(nodeIndex) != nodeStates.end()) {
      inputs = &nodeStates.at(nodeIndex).input;
    } else {
      auto index = traversedNodeStates.at(nodeIndex);
      inputs = &mTraversal.clusters()[index.clusterIndex]
                    .content[index.nodeStateIndex]
                    .input;
    }
    for (auto& input : *inputs) {
      if (input.nodeIndex == currentNodeState.nodeIndex) {
        input.clusterIndex = currentClusterIndex;
        input.nodeStateIndex = currentNodeStateIndex;
      }
    }
  }
}

const Traversal& GraphInternal::traverse() {
  if (mTraversal.isValidTraversal()) {
    return mTraversal;
  }
  std::sort(mTopology.begin(), mTopology.end());
  std::queue<utils::Index> currentQueue, newQueue;
  initQueue(currentQueue, mInputNodeIndexes);
  std::unordered_map<utils::Index, NodeState> nodeStates;
  std::unordered_map<utils::Index, NodeStateIndex> traversedNodeStates;
  initInputNodeStates(nodeStates);
  while (true) {
    Cluster cluster{0};
    while (!currentQueue.empty()) {
      size_t startNodeIndex = currentQueue.front();
      currentQueue.pop();
      Edge target(startNodeIndex, 0, 0);
      auto edgeIterator =
          std::lower_bound(mTopology.begin(), mTopology.end(), target);
      while (edgeIterator != mTopology.end() &&
             edgeIterator->startNodeIndex == startNodeIndex) {
        auto endNodeIndex = edgeIterator->endNodeIndex;
        auto getter = [](std::unordered_map<utils::Index, NodeState>& map,
                         utils::Index index) -> size_t& {
          return map[index].inputsCount;
        };
        auto resInputsCount =
            safetyIncrementMapValue(nodeStates, endNodeIndex, getter);
        auto targetInputCount = mInputsCount.at(endNodeIndex);
        if (resInputsCount == targetInputCount) {
          newQueue.push(endNodeIndex);
        } else if (resInputsCount > targetInputCount) {
          utils::FatalError(utils::ATH_FATAL_OTHER,
                            "traverse() in Graph: ", this,
                            ". Graph has cycle(s)");
        }
        nodeStates[startNodeIndex].output.emplace_back(endNodeIndex,
                                                       edgeIterator->mark);
        nodeStates[endNodeIndex].input.emplace_back(startNodeIndex,
                                                    edgeIterator->mark);
        nodeStates[endNodeIndex].isWayToFrozen =
            nodeStates[endNodeIndex].isWayToFrozen &&
            nodeStates[startNodeIndex].isWayToFrozen;
        ++edgeIterator;
      }
      auto& processedState = nodeStates[startNodeIndex];
      cluster.content.emplace_back(startNodeIndex, processedState.inputsCount,
                                   processedState.isWayToFrozen,
                                   std::move(processedState.input),
                                   std::move(processedState.output));
      traversedNodeStates[startNodeIndex] = NodeStateIndex{
          mTraversal.clusters().size(), cluster.content.size() - 1};
      nodeStates.erase(startNodeIndex);
      ++cluster.nodeCount;
      auto& currentNodeState = cluster.content.back();
      bypassDependenceOfCurrentNodeState(
          currentNodeState, mTraversal.clusters().size(),
          cluster.content.size() - 1, nodeStates, traversedNodeStates);
    }
    if (cluster.nodeCount > 0) {
      mTraversal.clusters().emplace_back(std::move(cluster));
    }
    std::swap(currentQueue, newQueue);
    if (currentQueue.empty()) {
      break;
    }
  }
  mTraversal.validTraversalFlag() = true;
  setUpTensors();
  return mTraversal;
}

utils::Index
GraphInternal::createInitialGradientNode(GraphInternal& gradientGraph, const NodeState* nodeStatePtr) const {
  auto context = mContext.lock();
  auto& node = context->getRef<AbstractNodeInternal>(nodeStatePtr->nodeIndex);
  auto tensor = node.getTensorPtr();
  auto loaderIndex = context->create<loaders::internal::ConstantLoaderInternal>(
      context, context->getNextPublicIndex(), 1.0);
  auto resultNodeIndex = context->create<InputNodeInternal>(
      context, context->getNextPublicIndex(), tensor->getShape(),
      tensor->getDataType(), true, loaderIndex,
      (std::string("InitialNode_") +
       std::to_string(context->getNextPublicIndex()))
          .data());
  gradientGraph.mInputNodeIndexes.emplace_back(resultNodeIndex);
  return resultNodeIndex;
}

void GraphInternal::mergeEdges(const std::vector<core::internal::Edge>& edges) {
  for (auto& edge : edges) {
    connect(edge.startNodeIndex, edge.endNodeIndex, edge.mark);
  }
}

utils::Index GraphInternal::accumulateOutputNodes(
    GraphInternal& gradient, const NodeState* nodeStatePtr,
    const std::unordered_map<const NodeState*, utils::Index>&
        mapNodeStateToFinalGradientIndex) const {
#ifdef DEBUG
  if (nodeStatePtr->output.size() == 0) {
    utils::FatalError(utils::ATH_ASSERT,
                      "Error while accumulateOutputNodes() is working. Output "
                      "of node state contains 0 states.");
  }
#endif
  auto context = mContext.lock();
  auto addOperationIndex =
      context->create<operation::internal::AddOperationInternal>(
          context, context->getNextPublicIndex(),
          (std::string("AddOperation_") +
           std::to_string(context->getNextPublicIndex()))
              .data());
  auto& shape =
      context->getRef<AbstractNodeInternal>(nodeStatePtr->nodeIndex)
          .getTensorPtr()
          ->getShape();
  auto zeroLoaderIndex =
      context->create<loaders::internal::ConstantLoaderInternal>(
          context, context->getNextPublicIndex(), 0.0,
          (std::string("ZeroLoaderIndex_") +
           std::to_string(context->getNextPublicIndex()))
              .data());
  auto finalGradientIndex = gradient.create<InputNodeInternal>(
      shape, DataType::FLOAT, true, zeroLoaderIndex,
      (std::string("ZeroInputNode_") +
       std::to_string(context->getNextPublicIndex()))
          .data());
  for (size_t indexOutputDependence = 0;
       indexOutputDependence < nodeStatePtr->output.size();
       ++indexOutputDependence) {
    auto& dependence = nodeStatePtr->output[indexOutputDependence];
    auto resNodeStatePtr = &mTraversal.getClusters()[dependence.clusterIndex]
                                .content[dependence.nodeStateIndex];
    auto& abstractNode =
        context->getRef<AbstractNodeInternal>(dependence.nodeIndex);
    if (abstractNode.getType() == NodeType::DEFAULT) {
      auto& node = static_cast<NodeInternal&>(abstractNode);
      auto operationPtr = node.getOperationPtr();
      auto [newFinalGradientIndex, edges, newInputNodes] =
          operationPtr->genDerivative(
              nodeStatePtr, resNodeStatePtr, indexOutputDependence,
              mapNodeStateToFinalGradientIndex.at(resNodeStatePtr));
      auto addNodeIndex = context->create<core::internal::NodeInternal>(
          context, context->getNextPublicIndex(), addOperationIndex,
          (std::string("AddNodeLinker_") +
           std::to_string(context->getNextPublicIndex()))
              .data());
      gradient.connect(newFinalGradientIndex, addNodeIndex,
                       operation::AddOperation::LEFT);
      gradient.connect(finalGradientIndex, addNodeIndex,
                       operation::AddOperation::RIGHT);
      finalGradientIndex = addNodeIndex;
      gradient.mergeEdges(edges);
      for (auto newInputNodeIndex : newInputNodes) {
        gradient.mInputNodeIndexes.emplace_back(newInputNodeIndex);
      }
    } else {
      // TODO error
    }
  }
  return finalGradientIndex;
}

std::tuple<utils::Index, std::unordered_map<utils::Index, utils::Index>>
GraphInternal::createGradientGraph(utils::Index targetNodeIndex) const {
  auto context = mContext.lock();
  auto& targetNode = context->getRef<AbstractNodeInternal>(targetNodeIndex);
  if (targetNode.getType() != NodeType::DEFAULT) {
    utils::FatalError(utils::ATH_BAD_ACCESS, "Target node isn't a functional node.");
    return {};
  }
  auto gradientGraphIndex = context->create<GraphInternal>(
      mContext, context->getNextPublicIndex(),
      (std::string("GradientGraph_") +
       std::to_string(context->getNextPublicIndex()))
          .data());
  auto& gradientGraph = context->getRef<GraphInternal>(gradientGraphIndex);
  std::unordered_map<const NodeState*, utils::Index>
      mapNodeStateToFinalGradientIndex;
  std::unordered_map<utils::Index, utils::Index> inputNodeChangers;
  if (mTraversal.getClusters().size() <= 1) {
    return {};
  }
  utils::Index gradientFinalNodeIndex = 0;
  bool targetNodeFound = false;
  size_t indexCluster = mTraversal.getClusters().size() - 1;
  auto& clusterCollection = mTraversal.getClusters();
  for (auto rClusterIterator = clusterCollection.rbegin(); rClusterIterator != clusterCollection.rend(); ++rClusterIterator) {
    auto& cluster = *rClusterIterator;
    for (const auto& nodeState : cluster.content) {
      if (!targetNodeFound) {
        if (nodeState.nodeIndex == targetNodeIndex) {
          targetNodeFound = true;
          gradientFinalNodeIndex = createInitialGradientNode(gradientGraph, &nodeState);
          mapNodeStateToFinalGradientIndex[&nodeState] = gradientFinalNodeIndex;
        }
      } else {
        if (nodeState.isWayToFrozen) {
          continue;
        }
        auto nodeStatePtr = &nodeState;
          gradientFinalNodeIndex = accumulateOutputNodes(
              gradientGraph, nodeStatePtr, mapNodeStateToFinalGradientIndex);
        mapNodeStateToFinalGradientIndex[nodeStatePtr] = gradientFinalNodeIndex;
        if (indexCluster == 0) {
          auto& inputNode = context->getRef<InputNodeInternal>(nodeState.nodeIndex);
          if (!inputNode.isFrozen()) {
            inputNodeChangers[inputNode.getPublicIndex()] = gradientFinalNodeIndex;
          }
        }
      }
    }
    --indexCluster;
  }
  gradientGraph.traverse();
  return std::make_tuple(gradientGraphIndex, inputNodeChangers);
}

utils::Index GraphInternal::createWeightChangingGraph(
    const std::unordered_map<utils::Index, utils::Index>& mapInputNodes) {
  auto context = mContext.lock();
  auto weightChangingGraphIndex = context->create<GraphInternal>(
      mContext, context->getNextPublicIndex(),
      (std::string("WeightChangingGraph_") +
       std::to_string(context->getNextPublicIndex()))
          .data());
  auto& weightChangingGraph =
      context->getRef<GraphInternal>(weightChangingGraphIndex);
  auto learningRateLoaderIndex =
      context->create<loaders::internal::ConstantLoaderInternal>(
          context, context->getNextPublicIndex(), -0.01,
          (std::string("LearningRateLoader_") +
           std::to_string(context->getNextPublicIndex()))
              .data()); // TODO give runtime args to graph
  auto dummyLoader =
      context->create<loaders::internal::DummyLoaderInternal>(
          context, context->getNextPublicIndex(),
          (std::string("DummyLoader_") +
              std::to_string(context->getNextPublicIndex()))
              .data());
  auto multiplyOperation =
      context->create<operation::internal::MulOperationInternal>(
          context, context->getNextPublicIndex(),
          (std::string("MultiplyOperation_") +
           std::to_string(context->getNextPublicIndex()))
              .data());
  auto addOperation =
      context->create<operation::internal::AddOperationInternal>(
          context, context->getNextPublicIndex(),
          (std::string("AddOperation_") +
           std::to_string(context->getNextPublicIndex()))
              .data());
  for (auto& edge : mapInputNodes) { /// grad - second, initial graph - first
    auto gradientFinalNodeIndex = edge.second;
    auto sourceGraphInputNodeIndex = edge.first;
    auto& sourceGraphInputNode =
        context->getRef<internal::AbstractNodeInternal>(
            sourceGraphInputNodeIndex);
    auto& gradientFinalNode =
        context->getRef<internal::AbstractNodeInternal>(gradientFinalNodeIndex);
    auto& gradientShape = gradientFinalNode.getTensorPtr()->getShape();
    auto gradientDataType = gradientFinalNode.getTensorPtr()->getDataType();
    auto dummyNodeIndex = context->create<InputNodeInternal>(context, context->getNextPublicIndex(), gradientShape, gradientDataType, true, dummyLoader, (std::string("DummyNodeGradValueHolder_") +
        std::to_string(context->getNextPublicIndex()))
        .data());
    auto& dummyNode = context->getRef<AbstractNodeInternal>(dummyNodeIndex);
    dummyNode.setTensorIndex(gradientFinalNode.getTensorIndex());
    auto learningRateHolderNodeIndex = context->create<InputNodeInternal>(
        context, context->getNextPublicIndex(), gradientShape, gradientDataType,
        true, learningRateLoaderIndex,
        (std::string("LearningRateHolderNode_") +
         std::to_string(context->getNextPublicIndex()))
            .data());
    weightChangingGraph.mInputNodeIndexes.emplace_back(dummyNodeIndex);
    weightChangingGraph.mInputNodeIndexes.emplace_back(learningRateHolderNodeIndex);
    auto multiplyNodeIndex = context->create<NodeInternal>(
        context, context->getNextPublicIndex(), multiplyOperation,
        (std::string("LearningRateMultiplyNode_") +
         std::to_string(context->getNextPublicIndex()))
            .data());
    weightChangingGraph.connect(dummyNodeIndex, multiplyNodeIndex,
                                operation::MulOperation::LEFT);
    weightChangingGraph.connect(learningRateHolderNodeIndex, multiplyNodeIndex,
                                operation::MulOperation::RIGHT);
    auto sourceGraphInputNodeHolderIndex = context->create<InputNodeInternal>(
        context, context->getNextPublicIndex(), gradientShape, gradientDataType,
        true, dummyLoader,
        (std::string("SourceHolderNode_") +
         std::to_string(context->getNextPublicIndex()))
            .data());
    auto& sourceGraphInputNodeHolder = context->getRef<AbstractNodeInternal>(sourceGraphInputNodeHolderIndex);
    sourceGraphInputNodeHolder.setTensorIndex(sourceGraphInputNode.getTensorIndex());
    weightChangingGraph.mInputNodeIndexes.emplace_back(sourceGraphInputNodeHolderIndex);
    auto addNodeIndex = context->create<NodeInternal>(
        context, context->getNextPublicIndex(), addOperation,
        (std::string("SourceNodeChanger_") +
            std::to_string(context->getNextPublicIndex()))
            .data());
    auto& addNode = context->getRef<NodeInternal>(addNodeIndex);
    addNode.setTensorIndex(sourceGraphInputNode.getTensorIndex());
    weightChangingGraph.connect(multiplyNodeIndex, addNodeIndex,
                                operation::AddOperation::LEFT);
    weightChangingGraph.connect(sourceGraphInputNodeHolderIndex,
                                addNodeIndex,
                                operation::AddOperation::RIGHT);
  }
  weightChangingGraph.traverse();
  return weightChangingGraphIndex;
}

std::tuple<utils::Index, utils::Index> GraphInternal::getGradient(utils::Index targetNodeIndex) {
  traverse();
  auto [gradientCalculatingGraphIndex, mapInputNodes] = createGradientGraph(targetNodeIndex);


  auto weightChangingGraphIndex = createWeightChangingGraph(mapInputNodes);
  return std::make_tuple(gradientCalculatingGraphIndex,
                         weightChangingGraphIndex);

//  return std::make_tuple(gradientCalculatingGraphIndex,
//                         0);
}

} // namespace athena::core::internal
