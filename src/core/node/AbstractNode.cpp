#include <utility>

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

#include <athena/core/AbstractNode.h>
#include <athena/core/Graph.h>

namespace athena::core {
AbstractNode::AbstractNode(const AbstractNode& rhs)
    : mResultTensor(rhs.mResultTensor),
      mOutgoingDerivatives(rhs.mOutgoingDerivatives),
      mOwnDerivativeTensor(rhs.mOwnDerivativeTensor),
      mContext(rhs.mContext),
      mName(rhs.mName),
      mGraphIndex(inner::kKUndefinedIndex),
      mNodeIndex(inner::getNodeTable(*mContext).registerRecord(this)),
      mInputsCount(0) {}
AbstractNode::AbstractNode(AbstractNode&& rhs) noexcept
    : mResultTensor(std::move(rhs.mResultTensor)),
      mOutgoingDerivatives(std::move(rhs.mOutgoingDerivatives)),
      mOwnDerivativeTensor(std::move(rhs.mOwnDerivativeTensor)),
      mContext(rhs.mContext),
      mName(std::move(rhs.mName)),
      mGraphIndex(rhs.mGraphIndex),
      mNodeIndex(rhs.mNodeIndex),
      mInputsCount(rhs.mInputsCount) {
    inner::getNodeTable(*mContext)[mNodeIndex] = this;
    rhs.fullClear();
}
AbstractNode::AbstractNode(TensorShape shape,
                           DataType dataType,
                           Context& context,
                           std::string name)
    : mResultTensor(
          std::make_shared<inner::Tensor>(dataType, std::move(shape), context)),
      mContext(&context),
      mName(std::move(name)),
      mGraphIndex(inner::kKUndefinedIndex),
      mNodeIndex(inner::getNodeTable(*mContext).registerRecord(this)),
      mInputsCount(0) {}
AbstractNode::~AbstractNode() {
    inner::getNodeTable(*mContext)[mNodeIndex] = nullptr;
}
void AbstractNode::fullClear() {
    AbstractNode::clear();
    mGraphIndex = inner::kKUndefinedIndex;
    mNodeIndex = inner::kKUndefinedIndex;
    mInputsCount = 0;
}
void AbstractNode::after(const AbstractNode& node, EdgeMark mark) const {
    if (auto* graph = inner::getGraphTable(*mContext)[mGraphIndex]; graph) {
        graph->link(node, *this, mark);
    } else {
        FatalError(ATH_BAD_ACCESS, "Graph which contains node ", this,
                   " does not exists");
    }
}
void AbstractNode::before(const AbstractNode& node, EdgeMark mark) const {
    if (auto* graph = inner::getGraphTable(*mContext)[mGraphIndex]; graph) {
        graph->link(*this, node, mark);
    } else {
        FatalError(ATH_BAD_ACCESS, "Graph which contains node ", this,
                   " does not exists");
    }
}
ShapeView AbstractNode::getShapeView() const {
    return mResultTensor->getShapeView();
}
ShapeView AbstractNode::getSubShapeView(size_t offset) const {
    return mResultTensor->getSubShapeView(offset);
}
const TensorShape& AbstractNode::getShape() const {
    return mResultTensor->getShape();
}
DataType AbstractNode::getDataType() const {
    return mResultTensor->getDataType();
}
NodeIndexType AbstractNode::getNodeIndex() const {
    return mNodeIndex;
}
size_t AbstractNode::getGraphIndex() const {
    return mGraphIndex;
}
size_t AbstractNode::getInputsCount() const {
    return mInputsCount;
}
std::string_view AbstractNode::getName() const {
    return mName;
}
std::string& AbstractNode::name() {
    return mName;
}
const std::string& AbstractNode::name() const {
    return mName;
}
void AbstractNode::setShape(const TensorShape& shape) {
    if (mGraphIndex != inner::kKUndefinedIndex) {
        FatalError(
            ATH_FATAL_OTHER,
            "It is forbidden to change shapes of nodes which belongs to graph");
    }
    mResultTensor->setShape(shape);
}
void AbstractNode::clear() {
    mResultTensor->clear();
    mName.clear();
}
void AbstractNode::removeFromGraph() {
    if (auto* graph = inner::getGraphTable(*mContext)[mGraphIndex]; graph) {
        graph->removeNode(*this);
    }
}
void AbstractNode::saveInGraph(bool isRepairedNode) {
    if (auto* graph = inner::getGraphTable(*mContext)[mGraphIndex]; graph) {
        graph->saveNode(*this, isRepairedNode);
    }
}
AbstractNode::AbstractNode(Context& context, std::string name)
    : mResultTensor(inner::getNullTensor(context)),
      mContext(&context),
      mName(std::move(name)),
      mGraphIndex(inner::kKUndefinedIndex),
      mNodeIndex(inner::getNodeTable(*mContext).registerRecord(this)),
      mInputsCount(0) {}
void inner::setResultTensor(
    athena::core::AbstractNode& node,
    std::shared_ptr<athena::core::inner::Tensor> tensor) {
    node.mResultTensor = std::move(tensor);
}
void inner::addOutgoingDerivative(AbstractNode& node, std::shared_ptr<inner::Tensor> tensor, size_t outgoingNodeIndex) {
    node.mOutgoingDerivatives[outgoingNodeIndex] = std::move(tensor);
}
inner::Tensor& inner::getOutgoingDerivative(AbstractNode& node, NodeIndexType index) {
    return *node.mOutgoingDerivatives[index];
}
inner::Tensor& inner::getOwnDerivative(AbstractNode& node) {
    return *node.mOwnDerivativeTensor;
}
std::map<size_t, std::shared_ptr<inner::Tensor>> &inner::getOutgoingDerivatives(AbstractNode &node) {
    return node.mOutgoingDerivatives;
}
}  // namespace athena::core
