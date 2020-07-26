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
#include <polarai/core/graph/Graph.hpp>

namespace polarai::core::internal {

class POLAR_CORE_EXPORT Executor {
public:
  Executor() = default;
  Executor(const Executor&) = default;
  Executor(Executor&&) = default;
  Executor& operator=(const Executor&) = default;
  Executor& operator=(Executor&&) = default;

  virtual ~Executor() = default;
  virtual void addGraph(Graph& graph) = 0;
  virtual void evaluate(Graph& graph) = 0;
};

} // namespace polarai::core::internal
