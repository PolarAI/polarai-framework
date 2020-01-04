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

#ifndef ATHENA_ABSTRACTNODE_H
#define ATHENA_ABSTRACTNODE_H

#include <athena/core/Context.h>
#include <athena/core/DataType.h>
#include <athena/core/NodeType.h>
#include <athena/core/core_export.h>
#include <athena/core/inner/InnerFunctions.h>
#include <athena/core/inner/Tensor.h>
#include <athena/core/inner/ForwardDeclarations.h>

#include <string>
#include <string_view>
#include <map>

namespace athena::core {


/**
 * A Node represents a piece of computation in Graph
 */
class ATH_CORE_EXPORT AbstractNode {
    private:
    void fullClear();
    std::shared_ptr<inner::Tensor> mResultTensor;
    std::map<size_t, std::shared_ptr<inner::Tensor>> mOutgoingDerivatives;      // Index of outgoing node -> outgoing derivative for that tensor
    std::shared_ptr<inner::Tensor> mOwnDerivativeTensor;
    Context* mContext;
    std::string mName;
    size_t mGraphIndex;
    size_t mNodeIndex;
    size_t mInputsCount;

    protected:
    friend void inner::addOutgoingDerivative(AbstractNode& node, std::shared_ptr<inner::Tensor> tensor, size_t outgoingNodeIndex);
    friend inner::Tensor& inner::getOutgoingDerivative(AbstractNode& node, NodeIndexType index);
    friend inner::Tensor& inner::getOwnDerivative(AbstractNode& node);
    friend std::map<size_t, std::shared_ptr<inner::Tensor>> &inner::getOutgoingDerivatives(AbstractNode &node);
    friend std::shared_ptr<inner::Tensor> inner::getTensorSmartPtrFromNode(AbstractNode &node);
    AbstractNode(TensorShape shape,
                 DataType dataType,
                 Context& context,
                 std::string name);

    public:
    AbstractNode() = delete;
    AbstractNode(const AbstractNode& rhs);
    AbstractNode(AbstractNode&& rhs) noexcept;

    AbstractNode(Context& context, std::string name);
    virtual ~AbstractNode();

    AbstractNode& operator=(const AbstractNode& rhs) = delete;
    AbstractNode& operator=(AbstractNode&& rhs) = delete;

    void after(const AbstractNode& node, EdgeMark mark) const;
    void before(const AbstractNode& node, EdgeMark mark) const;
    ShapeView getShapeView() const;
    ShapeView getSubShapeView(size_t offset = 1) const;
    const TensorShape& getShape() const;
    DataType getDataType() const;
    virtual NodeType getType() const = 0;
    NodeIndexType getNodeIndex() const;
    size_t getGraphIndex() const;
    size_t getInputsCount() const;
    std::string_view getName() const;
    std::string& name();
    const std::string& name() const;
    void setShape(const TensorShape& shape);
    void removeFromGraph();
    void saveInGraph(bool isRepairedNode = true);
    virtual void clear();
    friend void inner::setGraphIndex(AbstractNode& node, size_t graphIndex);
    friend void inner::incrementInputCount(athena::core::AbstractNode& node);
    friend inner::Tensor& inner::getTensorFromNode(AbstractNode& node);
    friend inner::Tensor* inner::getTensorPtrFromNode(AbstractNode& node);
    friend void inner::setResultTensor(
        athena::core::AbstractNode& node,
        std::shared_ptr<athena::core::inner::Tensor> tensor);
};
}  // namespace athena::core

#endif  // ATHENA_ABSTRACTNODE_H
