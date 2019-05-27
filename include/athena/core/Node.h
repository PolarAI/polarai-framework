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

#ifndef ATHENA_NODE_H
#define ATHENA_NODE_H

#include <athena/core/AbstractNode.h>
#include <athena/core/Operation.h>

namespace athena::core {
/**
 * Node holds Operation and Tensors to perform computation
 */
class Node : public AbstractNode {
    private:
    void fullClear();

    protected:
    Operation* mOperation;
    std::vector<inner::Tensor> mDerivativeTensors;

    friend void inner::addDerivativeTensor(Node& node, inner::Tensor& tensor);

    public:
    Node() = delete;
    Node(const Node& rhs) = default;
    Node(Node&& rhs) noexcept;
    Node(TensorShape shape,
         DataType dataType,
         Operation& operation,
         std::string name = "");
    ~Node() override;

    Node& operator=(const Node& rhs) = default;
    Node& operator=(Node&& rhs) noexcept;

    NodeType getType() const override;
    const Operation& getOperation() const;
    const Operation* getOperationPtr() const;
    Operation& operation();
    void setOperation(Operation& operation);
    void clear() override;
};
}  // namespace athena::core

#endif  // ATHENA_NODE
