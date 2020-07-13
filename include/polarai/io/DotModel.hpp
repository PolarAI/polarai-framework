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

#include <polarai/core/graph/Graph.hpp>
#include <polar_io_export.h>

#include <ostream>

namespace polarai::io {
/**
 * Print graph to DOT format for debug purposes
 */
class POLAR_IO_EXPORT DotModel {
public:
  static void exportGraph(core::Graph& graph, std::ostream& stream);
};
} // namespace polarai::model
