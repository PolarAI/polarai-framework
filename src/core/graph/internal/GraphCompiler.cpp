/*
 * Copyright (c) 2019 Athena. All rights reserved.
 * https://getathena.ml
 *
 * Licensed under MIT license.
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS, WITHOUT
 * WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the
 * License for the specific language governing permissions and limitations under
 * the License.
 */

#include <athena/core/graph/internal/GraphCompiler.h>

namespace athena::core::internal {
void GraphCompiler::compileForward(core::Graph& graph, Generator& generator) {}

void GraphCompiler::compileBackward(core::Graph& graph, Generator& generator) {}
//
// void GraphCompiler::setOptimizer(std::shared_ptr<Optimizer> optimizer) {
//  mOptimizer = std::move(optimizer);
//}
} // namespace athena::core::internal
