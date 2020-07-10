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

#ifndef ATHENA_BACKENDACCESSOR_H
#define ATHENA_BACKENDACCESSOR_H

#include <athena/core/tensor/Accessor.h>

#include <cstddef>
#include <cstdint>
#include <initializer_list>
#include <vector>

namespace athena::backend::llvm {
template <typename T> class BackendAccessor final : public core::Accessor<T> {
public:
  BackendAccessor(T* data, size_t dims, uint64_t* shape)
      : mData(data), mShape(shape, shape + dims){};
  auto operator()(std::initializer_list<size_t> idx) -> T& override {
    return mData[linearIndex(idx)];
  }

  auto operator()(size_t idx) -> T& override { return mData[idx]; }

  auto getShape() -> std::vector<size_t>& override { return mShape; }

  auto getRawPtr() -> T* override { return mData; }

private:
  T* mData;
  std::vector<size_t> mShape;

  auto linearIndex(std::initializer_list<size_t> idx) -> size_t {
    std::vector<size_t> unwrappedIdx{idx};

    size_t index = 0;
    size_t mul = 1;

    for (size_t i = 0; i != mShape.size(); ++i) {
      index += unwrappedIdx[i] * mul;
      mul *= mShape[i];
    }

    return index;
  }
};
} // namespace athena::backend::llvm

#endif // ATHENA_ACCESSOR_H
