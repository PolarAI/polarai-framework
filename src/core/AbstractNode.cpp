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
#include <athena/core/inner/GlobalTables.h>

namespace athena::core {
AbstractNode::AbstractNode(const AbstractNode& rhs)
    : mTensor(rhs.mTensor),
      mName(rhs.mName),
      mGraphIndex(inner::kKUndefinedIndex),
      mNodeIndex(inner::getNodeTable().registerRecord(this)),
      mInputsCount(0) {}
AbstractNode::AbstractNode(AbstractNode&& rhs) noexcept
    : mTensor(std::move(rhs.mTensor)),
      mName(std::move(rhs.mName)),
      mGraphIndex(rhs.mGraphIndex),
      mNodeIndex(rhs.mNodeIndex),
      mInputsCount(rhs.mInputsCount) {
    inner::getNodeTable()[mNodeIndex] = this;
    rhs.fullClear();
}
AbstractNode::AbstractNode(TensorShape shape,
                           DataType dataType,
                           std::string name)
    : mTensor(dataType, std::move(shape)),
      mName(std::move(name)),
      mGraphIndex(inner::kKUndefinedIndex),
      mNodeIndex(inner::getNodeTable().registerRecord(this)),
      mInputsCount(0) {}
AbstractNode::~AbstractNode() {
    inner::getNodeTable()[mNodeIndex] = nullptr;
}
AbstractNode& AbstractNode::operator=(const AbstractNode& rhs) {
    mTensor = rhs.mTensor;
    mName = rhs.mName;
    return *this;
}
AbstractNode& AbstractNode::operator=(AbstractNode&& rhs) noexcept {
    // saveInGraph(false);
    if (mGraphIndex != inner::kKUndefinedIndex) {
        FatalError(1, "Move into node, which belongs to graph");
    }
    mTensor = std::move(rhs.mTensor);
    mName = std::move(rhs.mName);
    mGraphIndex = rhs.mGraphIndex;
    mNodeIndex = rhs.mNodeIndex;
    mInputsCount = rhs.mInputsCount;
    inner::getNodeTable()[mNodeIndex] = this;
    rhs.fullClear();
    return *this;
}
void AbstractNode::fullClear() {
    AbstractNode::clear();
    mGraphIndex = inner::kKUndefinedIndex;
    mNodeIndex = inner::kKUndefinedIndex;
    mInputsCount = 0;
}
void AbstractNode::after(const AbstractNode& node, EdgeMark mark) const {
    if (auto* graph = inner::getGraphTable()[mGraphIndex]; graph) {
        graph->link(node, *this, mark);
    } else {
        FatalError(1, "Graph which contains node ", this, " does not exists");
    }
}
void AbstractNode::before(const AbstractNode& node, EdgeMark mark) const {
    if (auto* graph = inner::getGraphTable()[mGraphIndex]; graph) {
        graph->link(*this, node, mark);
    } else {
        FatalError(1, "Graph which contains node ", this, " does not exists");
    }
}
ShapeView AbstractNode::getShapeView() const {
    return mTensor.getShapeView();
}
ShapeView AbstractNode::getSubShapeView(size_t offset) const {
    return mTensor.getSubShapeView(offset);
}
DataType AbstractNode::getDataType() const {
    return mTensor.getDataType();
}
size_t AbstractNode::getNodeIndex() const {
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
void AbstractNode::setShape(const TensorShape& shape) {
    mTensor.setShape(shape);
}
void AbstractNode::clear() {
    mTensor.clear();
    mName.clear();
}
void AbstractNode::removeFromGraph() {
    if (auto* graph = inner::getGraphTable()[mGraphIndex]; graph) {
        graph->removeNode(*this);
    }
}
void AbstractNode::saveInGraph(bool isRepairedNode) {
    if (auto* graph = inner::getGraphTable()[mGraphIndex]; graph) {
        graph->saveNode(*this, isRepairedNode);
    }
}
}  // namespace athena::core
