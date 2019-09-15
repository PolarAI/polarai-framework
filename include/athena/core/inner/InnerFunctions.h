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

#include <athena/core/inner/ForwardDeclarations.h>
#include <athena/core/inner/Tensor.h>

#include <cstddef>

namespace athena::core {
class AbstractNode;
}

namespace athena::core::inner {
void setGraphIndex(AbstractNode &node, size_t graphIndex);
void incrementInputCount(AbstractNode &node);
Tensor &getTensorFromNode(AbstractNode &node);
Traversal &getTraversal(Graph &graph);
Clusters &getClusters(Graph &graph);
Clusters &getClusters(Traversal &traversal);
void setTraversalValidity(Traversal &traversal, bool flag);
void addDerivativeTensor(Node &node, inner::Tensor &tensor);
Tensor &getDerivativeTensor(Node &node, size_t index);
void addErrorTensor(Node &node, inner::Tensor &tensor);
Tensor &getErrorTensor(Node &node, size_t index);
void setResultTensor(AbstractNode &node, inner::Tensor *tensor);
Tensor *getNullTensor();
}  // namespace athena::core::inner

#endif  // ATHENA_INNERFUNCTIONS_H
