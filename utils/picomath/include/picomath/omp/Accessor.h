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

#ifndef ATHENA_ACCESSOR_H
#define ATHENA_ACCESSOR_H

#include "Index.h"

namespace {
constexpr size_t linearIndex(Index<1> idx, std::array<size_t, 1>) {
  return idx[0];
}
constexpr size_t linearIndex(Index<2> idx, std::array<size_t, 2> sizes) {
  return idx[1] + idx[0] * sizes[1];
}
constexpr size_t linearIndex(Index<3> idx, std::array<size_t, 3> sizes) {
  return idx[2] + idx[1] * sizes[2] + idx[0] * sizes[2] * sizes[1];
}
} // namespace

/// Provides access to shaped data.
///
/// The interface of this class is inspired by SYCL accessors and aims to be
/// backwards compatible in every way that is required by kernels.
///
/// No memory allocations are done inside this class. It only stores raw pointer
/// to data and its shape.
///
/// \tparam DataT is type of underlying data structure.
/// \tparam Dims is dimension of data. Can be 1, 2, or 3.
template <typename DataT, int Dims> class Accessor {
private:
  DataT* mData;
  const std::array<size_t, Dims> mSizes;

public:
  using value_type = DataT;

  Accessor(DataT* data, std::array<size_t, Dims> sizes)
      : mData(data), mSizes(sizes) {}

  std::array<size_t, Dims> get_range() { return mSizes; }

  DataT& operator[](Index<Dims> idx) { return mData[linearIndex(idx, mSizes)]; }
};

#endif // ATHENA_ACCESSOR_H
