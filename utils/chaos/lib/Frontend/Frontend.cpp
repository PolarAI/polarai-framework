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

#include "CXX/CXXFrontend.h"
#include <Frontend/Frontend.h>

namespace chaos {
std::vector<std::string> Frontend::run(std::string_view filename,
                                       std::vector<std::string> args) {
  CXXFrontend cxxFrontend;

  cxxFrontend.run(filename, args);

  std::vector<std::string> resultFiles;
  return resultFiles;
}
} // namespace chaos