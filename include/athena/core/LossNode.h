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

#ifndef ATHENA_LOSSNODE_H
#define ATHENA_LOSSNODE_H

#include <athena/core/Criterion.h>
#include <athena/core/Node.h>

namespace athena::core {
/**
 * Special type of node that use for backward propagation on graph
 */
class LossNode : public Node {
    private:
    Criterion mCriterion;

    public:
    LossNode() = delete;
    LossNode(const LossNode& rhs) = default;
    LossNode(LossNode&& rhs) noexcept;
    LossNode(Operation& operation,
             Criterion criterion,
             std::string name = "");
    ~LossNode() override;

    LossNode& operator=(const LossNode& rhs) = default;
    LossNode& operator=(LossNode&& rhs) noexcept;

    NodeType getType() const override;
    Criterion getCriterion() const;
    Criterion& criterion();
    const Criterion& criterion() const;
    void clear() override;
};
}  // namespace athena::core

#endif  // ATHENA_LOSSNODE_H
