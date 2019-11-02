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
#include <athena/core/core_export.h>
#include <athena/core/inner/ForwardDeclarations.h>

namespace athena::core {
template <typename TemplateNodeType>
NodeType ATH_CORE_EXPORT getNodeType() {
    new FatalError(1, "NodeType is not defined for given type");
    return NodeType::UNDEFINED;
}
template <>
NodeType ATH_CORE_EXPORT getNodeType<Node>() {
    return NodeType::DEFAULT;
}
template <>
NodeType ATH_CORE_EXPORT getNodeType<InputNode>() {
    return NodeType::INPUT;
}
template <>
NodeType ATH_CORE_EXPORT getNodeType<OutputNode>() {
    return NodeType::OUTPUT;
}
template <>
NodeType ATH_CORE_EXPORT getNodeType<LossNode>() {
    return NodeType::LOSS;
}
}  // namespace athena::core
