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
#include <type_traits>

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

constexpr size_t totalSize(std::array<size_t, 1> sizes) { return sizes[0]; }
constexpr size_t totalSize(std::array<size_t, 2> sizes) {
  return sizes[0] * sizes[1];
}
constexpr size_t totalSize(std::array<size_t, 3> sizes) {
  return sizes[0] * sizes[1] * sizes[2];
}

} // namespace

namespace picomath {

enum class access_mode { read, write };

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
template <typename DataT, int Dims, access_mode Mode,
          typename Allocator = std::allocator<DataT>>
class Accessor {
private:
  Allocator mAllocator;
  DataT* mData;
  const std::array<size_t, Dims> mSizes;

public:
  using value_type = DataT;

  explicit constexpr Accessor(std::array<size_t, Dims> sizes)
      : mData(mAllocator.allocate(totalSize(sizes))), mSizes(sizes) {}

  constexpr Accessor(DataT* data, std::array<size_t, Dims> sizes)
      : mData(data), mSizes(sizes) {}

  std::array<size_t, Dims> get_range() { return mSizes; }

  template <access_mode _Mode = Mode>
  std::enable_if_t<_Mode == access_mode::write, DataT&>
  constexpr operator[](Index<Dims> idx) {
    return mData[linearIndex(idx, mSizes)];
  }

  template <access_mode _Mode = Mode>
  std::enable_if_t<_Mode == access_mode::read, const DataT&>
  constexpr operator[](Index<Dims> idx) const {
    return mData[linearIndex(idx, mSizes)];
  }
};
} // namespace picomath
