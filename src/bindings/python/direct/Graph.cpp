//===----------------------------------------------------------------------===//
// Copyright (c) 2020 Athena. All rights reserved.
// https://getathena.ml
//
// Licensed under MIT license.
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS, WITHOUT
// WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the
// License for the specific language governing permissions and limitations under
// the License.
//===----------------------------------------------------------------------===//

#include "Graph.hpp"

#include <athena/core/graph/Graph.h>

namespace polar::python {
PyGraph::PyGraph(PyContext ctx, const std::string& name) {
  auto graph = ctx.getContext().create<athena::core::Graph>(name.data());
  mGraph = std::make_unique<athena::core::Graph>(ctx.getContext().internal(),
                                                 graph.getPublicIndex());
}
std::string PyGraph::getName() { return mGraph->getName().getString(); }
} // namespace polar::python
