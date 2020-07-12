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

#include <CL/sycl.hpp>

namespace AllocatorType {
enum AllocatorType : int { usm, buffer };
}

template <typename T, int Dims> class PointerAccessor {
private:
  T* mPointer;
  cl::sycl::range<Dims> mRange;

public:
  PointerAccessor(T* ptr, cl::sycl::range<Dims> range)
      : mPointer(ptr), mRange(range) {}

  auto operator[](cl::sycl::id<Dims> id) -> T& {
    if constexpr (Dims == 1) {
      return mPointer[id.get(0)];
    } else {
      size_t idx = 0;
      for (int i = 0; i < Dims; i++) {
        idx = idx * mRange[i] + id[i];
      }
      return mPointer[idx];
    }
  }

  auto operator[](size_t idx) -> T& { return mPointer[idx]; }

  auto get_pointer() -> cl::sycl::global_ptr<T> {
    return cl::sycl::global_ptr<T>(mPointer);
  }

  auto get_range() -> const cl::sycl::range<Dims>& {
    return mRange;
  } 
};

template <int Type, typename T, int Dims> struct read_accessor {};

template <typename T, int Dims>
struct read_accessor<AllocatorType::usm, T, Dims> {
  using type = PointerAccessor<const T, Dims>;
};

template <typename T, int Dims>
struct read_accessor<AllocatorType::buffer, T, Dims> {
  using type = cl::sycl::accessor<T, Dims, cl::sycl::access::mode::read,
                                  cl::sycl::access::target::global_buffer>;
};

template <int Type, typename T, int Dims>
struct read_write_accessor {};

template <typename T, int Dims>
struct read_write_accessor<AllocatorType::usm, T, Dims> {
  using type = PointerAccessor<T, Dims>;
};

template <typename T, int Dims>
struct read_write_accessor<AllocatorType::buffer, T, Dims> {
  using type = cl::sycl::accessor<T, Dims, cl::sycl::access::mode::read_write,
                                  cl::sycl::access::target::global_buffer>;
};

template <int Type, typename T, int Dims>
struct discard_write_accessor {};

template <typename T, int Dims>
struct discard_write_accessor<AllocatorType::usm, T, Dims> {
  using type = PointerAccessor<T, Dims>;
};

template <typename T, int Dims>
struct discard_write_accessor<AllocatorType::buffer, T, Dims> {
  using type =
      cl::sycl::accessor<T, Dims, cl::sycl::access::mode::discard_write,
                         cl::sycl::access::target::global_buffer>;
};

template <int AllocT, typename T, int Dims>
using read_accessor_t = typename read_accessor<AllocT, T, Dims>::type;

template <int AllocT, typename T, int Dims>
using read_write_accessor_t =
    typename read_write_accessor<AllocT, T, Dims>::type;

template <int AllocT, typename T, int Dims>
using discard_write_accessor_t =
    typename discard_write_accessor<AllocT, T, Dims>::type;
