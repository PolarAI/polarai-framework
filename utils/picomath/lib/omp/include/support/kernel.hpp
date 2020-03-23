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

#pragma once

#include "Index.hpp"

namespace picomath {

namespace inner {
template <typename KernelT> void runKernel2D(KernelT kernel, Index<2> size) {
#pragma omp parallel for collapse(2)
  for (size_t x = 0; x < size[0]; x++) {
    for (size_t y = 0; y < size[1]; y++) {
      kernel(Index<2>{x, y});
    }
  }
}
} // namespace inner

template <typename KernelT, int Dims>
void runKernel(KernelT kernel, Index<Dims> indexSpaceSize) {
  if constexpr (Dims == 2) {
    inner::runKernel2D(std::move(kernel), indexSpaceSize);
  }
}

} // namespace picomath
