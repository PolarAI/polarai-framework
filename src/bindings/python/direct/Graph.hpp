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

#pragma once

#include "Context.hpp"

#include <athena/core/graph/Graph.h>
#include <athena/utils/Index.h>

namespace polar::python {
class PyGraph {
public:
  PyGraph(PyContext ctx, const std::string& name);

  std::string getName();

private:
  std::unique_ptr<athena::core::Graph> mGraph;
};
} // namespace polar::python
