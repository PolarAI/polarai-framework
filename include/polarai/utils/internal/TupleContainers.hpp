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

#pragma once

#include <polarai/utils/Index.hpp>

#include <tuple>

namespace polarai::utils::internal {
template <template <typename, typename...> typename Map,
          template <typename> typename Value, typename... Args>
struct TupleMaps {
  using Holder = std::tuple<Map<Index, Value<Args>>...>;
};

template <template <typename, typename...> typename Container, typename... Args>
struct TupleContainers {
  using Holder = std::tuple<Container<Args>...>;
};
} // namespace polarai::utils::internal
