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

#ifndef ATHENA_INNERFUNCTIONS_H
#define ATHENA_INNERFUNCTIONS_H

#include <athena/core/core_export.h>
#include <athena/core/inner/ForwardDeclarations.h>
#include <athena/core/inner/Tensor.h>

#include <cstddef>
#include <vector>
#include <memory>

namespace athena::core::inner {
ATH_CORE_EXPORT void setGraphIndex(AbstractNode &node, size_t graphIndex);
ATH_CORE_EXPORT void incrementInputCount(AbstractNode &node);
ATH_CORE_EXPORT Tensor &getTensorFromNode(AbstractNode &node);
ATH_CORE_EXPORT Tensor* getTensorPtrFromNode(AbstractNode &node);
ATH_CORE_EXPORT std::shared_ptr<Tensor> getTensorSmartPtrFromNode(AbstractNode &node);
ATH_CORE_EXPORT Traversal &getTraversal(Graph &graph);
ATH_CORE_EXPORT Clusters &getClusters(Graph &graph);
ATH_CORE_EXPORT Clusters &getClusters(Traversal &traversal);
ATH_CORE_EXPORT void setTraversalValidity(Traversal &traversal, bool flag);
ATH_CORE_EXPORT void addOutgoingDerivative(AbstractNode &node, std::shared_ptr<inner::Tensor> tensor, size_t outgoingNodeIndex);
ATH_CORE_EXPORT Tensor &getOutgoingDerivative(AbstractNode &node, athena::core::NodeIndexType index);
ATH_CORE_EXPORT inner::Tensor &getOwnDerivative(AbstractNode &node);
ATH_CORE_EXPORT void setResultTensor(AbstractNode &node,
                                     std::shared_ptr<inner::Tensor> tensor);
ATH_CORE_EXPORT Tensor *getNullTensor(Context &context);
ATH_CORE_EXPORT inner::Table<AllocationRecord> &getAllocationTable(
    athena::core::Context &context);
ATH_CORE_EXPORT inner::Table<Graph *> &getGraphTable(
    athena::core::Context &context);
ATH_CORE_EXPORT inner::Table<AbstractNode *> &getNodeTable(
    athena::core::Context &context);
ATH_CORE_EXPORT Context &getContext(athena::core::Graph &graph);
ATH_CORE_EXPORT std::map<size_t, std::shared_ptr<inner::Tensor>> &getOutgoingDerivatives(AbstractNode &node);
}  // namespace athena::core::inner

#endif  // ATHENA_INNERFUNCTIONS_H
