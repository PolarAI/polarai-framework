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

#include <Frontend/export.h>
#include <string>
#include <string_view>
#include <vector>

namespace chaos {
class CHAOS_FRONTEND_EXPORT Frontend {
public:
  std::vector<std::string> run(const std::vector<std::string>& args);
};
} // namespace chaos

#endif // ATHENA_FRONTEND_H
