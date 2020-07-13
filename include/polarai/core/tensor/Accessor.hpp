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

#include <cstddef>
#include <cstdint>
#include <initializer_list>
#include <vector>

namespace polarai::core {
template <typename T> class Accessor {
public:
  virtual auto operator()(std::initializer_list<size_t> idx) -> T& = 0;
  virtual auto operator()(size_t idx) -> T& = 0;

  virtual auto getShape() -> const std::vector<size_t>& = 0;

  virtual auto getRawPtr() -> T* = 0;

protected:
  auto linearIndex(std::initializer_list<size_t> idx,
                   const std::vector<size_t>& shape) -> size_t {
    std::vector<size_t> unwrappedIdx{idx};

    size_t index = 0;
    size_t mul = 1;

    for (size_t i = 0; i != shape.size(); ++i) {
      index += unwrappedIdx[i] * mul;
      mul *= shape[i];
    }

    return index;
  }
};
} // namespace polarai::core
