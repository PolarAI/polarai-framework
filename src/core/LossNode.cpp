/*
 * Copyright (c) 2019 Athena. All rights reserved.
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

#include <athena/core/LossNode.h>

namespace athena::core {
LossNode::LossNode(Operation& operation,
                   Criterion criterion,
                   std::string name)
    : Node(operation, std::move(name)),
      mCriterion(criterion) {}
LossNode::LossNode(LossNode&& rhs) noexcept
    : Node(std::move(rhs)), mCriterion(rhs.mCriterion) {
    rhs.mCriterion = Criterion::UNDEFINED;
}
LossNode::~LossNode() {
    saveInGraph(false);
}
LossNode& LossNode::operator=(LossNode&& rhs) noexcept {
    mCriterion = rhs.mCriterion;
    rhs.mCriterion = Criterion::UNDEFINED;
    Node::operator=(std::move(rhs));
    return *this;
}
NodeType LossNode::getType() const {
    return NodeType::LOSS;
}
void LossNode::clear() {
    mCriterion = Criterion::UNDEFINED;
    Node::clear();
}
Criterion LossNode::getCriterion() const {
    return mCriterion;
}
Criterion& LossNode::criterion() {
    return mCriterion;
}
const Criterion& LossNode::criterion() const {
    return mCriterion;
}
}  // namespace athena::core
