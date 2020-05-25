/*
 * Copyright (c) 2018 Athena. All rights reserved.
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

#ifndef ATHENA_UTILS_H
#define ATHENA_UTILS_H

#include <athena/utils/Index.h>
#include <iostream>
#include <set>

namespace athena::tests::unit {
template <typename Type, template <typename, typename...> typename Container>
void showContainer(std::ostream& stream, const Container<Type>& set,
                   const char* message) {
  stream << message << std::endl;
  for (auto& val : set) {
    stream << val << std::endl;
  }
  stream << std::endl;
}
}; // namespace athena::tests::unit

#endif // ATHENA_UTILS_H
