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

#ifndef ATHENA_UTIL_H
#define ATHENA_UTIL_H

#include <variant>

namespace chaos::impl {
template <typename T, size_t idx = std::variant_size_v<T> - 1>
constexpr typename std::enable_if<idx == 0, void>::type
dump(const T& v, size_t tab, bool isFirst) {
  std::get<idx>(v).dump(tab, isFirst);
}

template <typename T, size_t idx = std::variant_size_v<T> - 1>
constexpr typename std::enable_if<idx != 0, void>::type
dump(const T& v, size_t tab, bool isFirst) {
  if (v.index() == idx) {
    std::get<idx>(v).dump(tab, isFirst);
  } else {
    impl::dump<T, idx - 1>(v, tab, isFirst);
  }
}
} // namespace chaos::impl

#endif // ATHENA_UTIL_H
