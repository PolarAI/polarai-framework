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

#include <map>
#include <string>
#include <vector>

namespace polarai::backend::generic {
struct KernelDesc {
  size_t globalX;
  size_t globalY;
  size_t globalZ;
  size_t localX;
  size_t localY;
  size_t localZ;
};
struct ProgramDesc {
  enum class Type { PTX, SPIRV_SHADER };

  Type type;
  std::string target;
  std::vector<char> data;
  std::map<std::string, KernelDesc> kernels;
};
} // namespace polarai::backend::generic
