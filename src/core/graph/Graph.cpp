//===----------------------------------------------------------------------===//
// Copyright (c) 2020 PolarAI. All rights reserved.
//
// Licensed under MIT license.
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS, WITHOUT
// WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the
// License for the specific language governing permissions and limitations under
// the License.
//===----------------------------------------------------------------------===//

#include <polarai/core/Wrapper.hpp>
#include <polarai/core/graph/Graph.hpp>
#include <polarai/core/graph/internal/GraphInternal.hpp>

#include <iostream>

namespace polarai::core {
Graph::Graph(utils::SharedPtr<internal::ContextInternal> contextInternal,
             utils::Index publicGraphIndex)
    : PublicEntity(std::move(contextInternal), publicGraphIndex) {}

Graph::~Graph() {}

void Graph::connect(utils::Index startNode, utils::Index endNode,
                    EdgeMark edgeMark = 0) {
  mContext->getRef<internal::GraphInternal>(mPublicIndex)
      .connect(startNode, endNode, edgeMark);
}

const internal::GraphInternal* Graph::getGraphInternal() const {
  return mContext->getPtr<internal::GraphInternal>(mPublicIndex);
}

internal::GraphInternal* Graph::getGraphInternal() {
  return mContext->getPtr<internal::GraphInternal>(mPublicIndex);
}

utils::StringView Graph::getName() const {
  return getGraphInternal()->getName();
}

std::tuple<Graph, Graph> Graph::getGradient(utils::Index targetNodeIndex) {
  auto internalTuple = getGraphInternal()->getGradient(targetNodeIndex);
  return std::make_tuple(Graph(mContext, std::get<0>(internalTuple)),
                         Graph(mContext, std::get<1>(internalTuple)));
}

const Traversal& Graph::traverse() { return getGraphInternal()->traverse(); }
} // namespace polarai::core
