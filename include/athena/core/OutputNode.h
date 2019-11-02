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

#ifndef ATHENA_OUTPUTNODE_H
#define ATHENA_OUTPUTNODE_H

#include <athena/core/AbstractNode.h>
#include <athena/core/Accessor.h>
#include <athena/core/core_export.h>

namespace athena::core {
/**
 * Special type of Node that use for output of data
 */
class ATH_CORE_EXPORT OutputNode : public AbstractNode {
    public:
    OutputNode() = delete;
    OutputNode(const OutputNode& rhs) = default;
    OutputNode(OutputNode&& rhs) = default;
    explicit OutputNode(DataType dataType,
                        Context& context,
                        std::string name = "");
    ~OutputNode() override;

    OutputNode& operator=(const OutputNode& rhs) = delete;
    OutputNode& operator=(OutputNode&& rhs) = delete;

    NodeType getType() const override;

    template <typename T>
    Accessor<T> getAccessor(Allocator& allocator) {
        return Accessor<T>(allocator, inner::getTensorFromNode(*this), {});
    }
};
}  // namespace athena::core

#endif  // ATHENA_OUTPUTNODE_H
