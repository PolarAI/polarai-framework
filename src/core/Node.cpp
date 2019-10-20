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

#include <athena/core/Node.h>

namespace athena::core {
Node::Node(Node&& rhs) noexcept
    : AbstractNode(std::move(rhs)),
      mOperation(rhs.mOperation),
      mDerivativeTensors(std::move(rhs.mDerivativeTensors)),
      mErrorTensor(std::move(rhs.mErrorTensor)) {
    rhs.mOperation = nullptr;
}
Node::Node(Operation& operation, std::string name)
    : AbstractNode(std::move(name)), mOperation(&operation) {}
Node::~Node() {
    saveInGraph(false);
}
Node& Node::operator=(Node&& rhs) noexcept {
    mOperation = rhs.mOperation;
    rhs.mOperation = nullptr;
    mDerivativeTensors = std::move(rhs.mDerivativeTensors);
    AbstractNode::operator=(std::move(rhs));
    return *this;
}
NodeType Node::getType() const {
    return NodeType::DEFAULT;
}
const Operation& Node::getOperation() const {
    return *mOperation;
}
const Operation* Node::getOperationPtr() const {
    return mOperation;
}
Operation& Node::operation() {
    return *mOperation;
}
const Operation& Node::operation() const {
    return *mOperation;
}
void Node::setOperation(Operation& operation) {
    mOperation = &operation;
}
void Node::clear() {
    mOperation = nullptr;
    AbstractNode::clear();
}
void inner::addDerivativeTensor(Node& node, inner::Tensor& tensor) {
    node.mDerivativeTensors.push_back(&tensor);
}
inner::Tensor& inner::getDerivativeTensor(Node& node, size_t index) {
    return *node.mDerivativeTensors[index];
}
void inner::setErrorTensor(Node& node, Tensor* tensor) {
    node.mErrorTensor = tensor;
}
inner::Tensor& inner::getErrorTensor(Node& node) {
    return *node.mErrorTensor;
}
}  // namespace athena::core