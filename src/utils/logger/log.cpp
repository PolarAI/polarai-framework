/*
 * Copyright (c) 2019 Athena. All rights reserved.
 * https://getathena.ml
 *
 * Licensed under MIT license.
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an “AS IS” BASIS, WITHOUT
 * WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the
 * License for the specific language governing permissions and limitations under
 * the License.
 */

#include <athena/utils/logger/log.h>

namespace athena::utils {
const std::unique_ptr<LogHolder> logHolder = std::unique_ptr<LogHolder>();

AbstractLogger& log() {
  static Logger logger = Logger(std::cout);
  return logger;
}

AbstractLogger& err() {
  static Logger logger = Logger(std::cerr);
  // std::cout << "@@@@ " << static_cast<void*>(logHolder.get()) << std::endl;
  return logger;
}
} // namespace athena::utils
