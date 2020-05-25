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

#include <athena/core/Generator.h>
#include <athena/core/context/Context.h>
#include <athena/core/core_export.h>
#include <athena/core/graph/Graph.h>
#include <athena/utils/Index.h>
#include <athena/utils/Pointer.h>

namespace athena::core::internal {
class ATH_CORE_EXPORT GraphCompiler {
public:
  static void compileForward(Graph& graph, Generator& generator);
  static void compileBackward(Graph& graph, Generator& generator);
};
} // namespace athena::core::internal

#endif // ATHENA_GRAPHCOMPILER_H
