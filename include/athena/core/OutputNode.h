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

namespace athena::core {
/**
* Special type of Node that use for output of data
*/
class OutputNode : public AbstractNode {
public:
    OutputNode() = delete;
    OutputNode(const OutputNode& rhs) = default;
    OutputNode(OutputNode&& rhs) = default;
    explicit OutputNode(DataType dataType,
               std::string name = "");
    ~OutputNode() override;

    OutputNode& operator=(const OutputNode& rhs) = default;
    OutputNode& operator=(OutputNode&& rhs) noexcept = default;

    NodeType getType() const override;
};
}  // namespace athena::core

#endif //ATHENA_OUTPUTNODE_H
