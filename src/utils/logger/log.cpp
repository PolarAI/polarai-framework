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

#include <polarai/utils/logger/log.hpp>

namespace polarai::utils {
const std::unique_ptr<LogHolder> logHolder = std::unique_ptr<LogHolder>();

AbstractLogger& log() {
  static Logger logger = Logger(std::cout);
  return logger;
}

AbstractLogger& err() {
  static Logger logger = Logger(std::cerr);
  return logger;
}
} // namespace polarai::utils
