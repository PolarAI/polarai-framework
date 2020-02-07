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

#include <utility>

namespace chaos {
std::vector<std::string> Frontend::run(const std::string& fileName) {
  mCXXFrontend->run(fileName);

  std::vector<std::string> resultFiles;
  return resultFiles;
}
Frontend::Frontend(std::shared_ptr<DriverOptions> opts)
    : mOptions(std::move(opts)),
      mCXXFrontend(std::make_unique<CXXFrontend>(mOptions)) {}
} // namespace chaos