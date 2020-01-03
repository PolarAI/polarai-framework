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

namespace athena::core::inner {
ATH_CORE_EXPORT void setGraphIndex(AbstractNode& node, size_t graphIndex);
ATH_CORE_EXPORT void incrementInputCount(AbstractNode& node);
ATH_CORE_EXPORT Tensor& getTensorFromNode(AbstractNode& node);
ATH_CORE_EXPORT std::shared_ptr<Tensor>
getTensorPtrFromNode(AbstractNode& node);
ATH_CORE_EXPORT Traversal& getTraversal(Graph& graph);
ATH_CORE_EXPORT Clusters& getClusters(Graph& graph);
ATH_CORE_EXPORT Clusters& getClusters(Traversal& traversal);
ATH_CORE_EXPORT void setTraversalValidity(Traversal& traversal, bool flag);
ATH_CORE_EXPORT void addDerivativeTensor(Node& node, inner::Tensor& tensor);
ATH_CORE_EXPORT Tensor& getDerivativeTensor(Node& node, size_t index);
ATH_CORE_EXPORT void setErrorTensor(Node& node, Tensor* tensor);
ATH_CORE_EXPORT inner::Tensor& getIncomingGradient(Node& node);
ATH_CORE_EXPORT void setResultTensor(AbstractNode& node,
                                     std::shared_ptr<inner::Tensor> tensor);
ATH_CORE_EXPORT Tensor* getNullTensor(Context& context);
ATH_CORE_EXPORT inner::Table<AllocationRecord>&
getAllocationTable(athena::core::Context& context);
ATH_CORE_EXPORT inner::Table<Graph*>&
getGraphTable(athena::core::Context& context);
ATH_CORE_EXPORT inner::Table<AbstractNode*>&
getNodeTable(athena::core::Context& context);
ATH_CORE_EXPORT Context& getContext(athena::core::Graph& graph);
} // namespace athena::core::inner

#endif // ATHENA_INNERFUNCTIONS_H
