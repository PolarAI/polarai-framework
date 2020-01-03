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

#ifndef ATHENA_GRAPHCOMPILER_H
#define ATHENA_GRAPHCOMPILER_H

#include <athena/core/AbstractGenerator.h>
#include <athena/core/Graph.h>
#include <athena/core/core_export.h>

namespace athena::core {
class ATH_CORE_EXPORT GraphCompiler {
public:
  static void compileForward(Graph& graph, AbstractGenerator& generator);
  static void compileBackward(Graph& graph, AbstractGenerator& generator);

  template <typename T>
  using ClusterContainer = std::vector<core::inner::NodeDependencies<T>>;
};
} // namespace athena::core

#endif // ATHENA_GRAPHCOMPILER_H
