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
#include <athena/core/Logger.h>

namespace athena::core {
class Error;
AbstractLogger& Logger::streamImpl(const std::string& data) {
  mOutStream << data;
  return *this;
}
AbstractLogger& Logger::streamImpl(const char* data) {
  mOutStream << data;
  return *this;
}
AbstractLogger& Logger::streamImpl(const std::string_view& data) {
  mOutStream << data;
  return *this;
}
AbstractLogger& Logger::streamImpl(const Error& data) {
  mOutStream << data;
  return *this;
}
} // namespace athena::core