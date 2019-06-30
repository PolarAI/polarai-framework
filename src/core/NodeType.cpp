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

#include <athena/core/FatalError.h>
#include <athena/core/NodeType.h>
#include <athena/core/inner/ForwardDeclarations.h>

namespace athena::core {
template <typename TemplateNodeType>
NodeType getNodeType() {
    FatalError(1, "NodeType is not defined for given type");
    return NodeType::UNDEFINED;
}
template <>
NodeType getNodeType<Node>() {
    return NodeType::DEFAULT;
}
template <>
NodeType getNodeType<InputNode>() {
    return NodeType::INPUT;
}
template <>
NodeType getNodeType<OutputNode>() {
    return NodeType::OUTPUT;
}
template <>
NodeType getNodeType<LossNode>() {
    return NodeType::LOSS;
}
}  // namespace athena::core
