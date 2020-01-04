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

#include <athena/core/AbstractNode.h>
#include <athena/core/Graph.h>

namespace athena::core::inner {
void setGraphIndex(AbstractNode &node, size_t graphIndex) {
    node.mGraphIndex = graphIndex;
}
void incrementInputCount(AbstractNode &node) {
    ++node.mInputsCount;
}
Tensor &getTensorFromNode(AbstractNode &node) {
    return *node.mResultTensor;
}
Tensor* getTensorPtrFromNode(AbstractNode &node) {
    return node.mResultTensor.get();
}
std::shared_ptr<Tensor> getTensorSmartPtrFromNode(AbstractNode &node) {
    return node.mResultTensor;
}
Traversal &getTraversal(Graph &graph) {
    return graph.mTraversal;
}
Clusters &getClusters(Graph &graph) {
    return getClusters(getTraversal(graph));
}
Clusters &getClusters(Traversal &traversal) {
    return traversal.clusters;
}
void setTraversalValidity(Traversal &traversal, bool flag) {
    traversal.mIsValidTraversal = flag;
}
}  // namespace athena::core::inner