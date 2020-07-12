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

#pragma once

#include <polar_core_export.h>
#include <polarai/core/Generator.hpp>
#include <polarai/core/context/Context.hpp>
#include <polarai/core/graph/Graph.hpp>
#include <polarai/utils/Index.hpp>
#include <polarai/utils/Pointer.hpp>

namespace polarai::core::internal {
class POLAR_CORE_EXPORT GraphCompiler {
public:
  static void compile(Graph& graph, Generator& generator);
};
} // namespace polarai::core::internal
