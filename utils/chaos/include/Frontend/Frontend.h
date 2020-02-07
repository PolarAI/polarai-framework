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

#ifndef ATHENA_FRONTEND_H
#define ATHENA_FRONTEND_H

#include <Driver/DriverOptions.h>
#include <Frontend/export.h>

#include <string>
#include <vector>

namespace chaos {
class CXXFrontend;
class CHAOS_FRONTEND_EXPORT Frontend {
private:
  std::shared_ptr<DriverOptions> mOptions;
  std::shared_ptr<CXXFrontend> mCXXFrontend;

public:
  Frontend(std::shared_ptr<DriverOptions> opts);
  std::vector<std::string> run(const std::string& fileName);
};
} // namespace chaos

#endif // ATHENA_FRONTEND_H
