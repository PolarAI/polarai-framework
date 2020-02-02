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
#include "Python/PythonFrontend.cpp"
#include <Frontend/Frontend.h>

namespace chaos {

enum class FEKind { Undefined, CXX, Python };

static bool hasEnding(std::string const& fullString,
                      std::string const& ending) {
  if (fullString.length() >= ending.length()) {
    return (0 == fullString.compare(fullString.length() - ending.length(),
                                    ending.length(), ending));
  } else {
    return false;
  }
}

std::vector<std::string> Frontend::run(const std::vector<std::string>& args) {
  CXXFrontend cxxFrontend;
  PythonFrontend pythonFrontend;

  FEKind kind = FEKind::Undefined;

  for (auto& arg : args) {
    if (hasEnding(arg, ".cpp")) {
      kind = FEKind::CXX;
      break;
    } else if (hasEnding(arg, ".py")) {
      kind = FEKind::Python;
      break;
    }
  }

  switch (kind) {
  case FEKind::CXX:
    cxxFrontend.run(args);
    break;
  case FEKind::Python:
    pythonFrontend.run(args);
    break;
  default:
    exit(-1);
  }

  std::vector<std::string> resultFiles;
  return resultFiles;
}
} // namespace chaos
