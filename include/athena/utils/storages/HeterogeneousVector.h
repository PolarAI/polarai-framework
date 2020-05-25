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

#ifndef ATHENA_HETEROGENEOUSVECTOR_H
#define ATHENA_HETEROGENEOUSVECTOR_H

#include <athena/utils/Index.h>
#include <athena/utils/allocator/Allocator.h>

#include <cstring>
#include <iostream>

namespace athena::utils {
using OffsetType = unsigned short;

template <typename BaseClass> class HeterogeneousVector {
public:
  // TODO doesn't work if capacity is 0
  explicit HeterogeneousVector(size_t capacity, size_t elementAverageSize,
                               Allocator allocator = Allocator());

  ~HeterogeneousVector();

  HeterogeneousVector(HeterogeneousVector&&);

  template <typename Type, typename... Args>
  utils::Index emplaceBack(Args&&... args);

  [[nodiscard]] size_t size() const;

  template <typename Type> Type& get(size_t index);

  template <typename Type> const Type& get(size_t index) const;

  template <typename Type> bool isAbleToContain() const;

  size_t capacity() const;

  size_t elementAverageSize() const;

  size_t calculateRequiredCapacity(size_t addedElementSize);

private:
  void destroy(bool isDestruct);

  void resize(size_t capacity = 0);

  Allocator mAllocator;
  size_t mElementAverageSize;
  size_t mNextIndex;
  size_t mCapacity;
  OffsetType* mOffsets;
  byte* mBytes;
};

template <typename BaseClass>
HeterogeneousVector<BaseClass>::HeterogeneousVector(
    HeterogeneousVector<BaseClass>&& rhs)
    : mAllocator(rhs.mAllocator), mElementAverageSize(rhs.mElementAverageSize),
      mNextIndex(rhs.mNextIndex), mCapacity(rhs.mCapacity),
      mOffsets(rhs.mOffsets), mBytes(rhs.mBytes) {
  rhs.mElementAverageSize = 0;
  rhs.mNextIndex = 0;
  rhs.mCapacity = 0;
  rhs.mOffsets = nullptr;
  rhs.mBytes = nullptr;
}

template <typename BaseClass>
template <typename Type, typename... Args>
utils::Index HeterogeneousVector<BaseClass>::emplaceBack(Args&&... args) {
  auto requiredCapacity = calculateRequiredCapacity(sizeof(Type));
  resize(requiredCapacity);
  if (mNextIndex == 0) {
    mOffsets[mNextIndex + 1] = sizeof(Type);
  } else {
    mOffsets[mNextIndex + 1] = mOffsets[mNextIndex] + sizeof(Type);
  }
  auto resIndex = mNextIndex++;
  new (static_cast<unsigned char*>(mBytes) + mOffsets[resIndex])
      Type(std::forward<Args>(args)...);
  return resIndex;
}

template <typename BaseClass>
HeterogeneousVector<BaseClass>::HeterogeneousVector(size_t capacity,
                                                    size_t elementAverageSize,
                                                    Allocator allocator)
    : mAllocator(std::move(allocator)), mElementAverageSize(elementAverageSize),
      mCapacity(0), mNextIndex(0), mOffsets(nullptr), mBytes(nullptr) {
  resize(capacity);
}

template <typename BaseClass>
HeterogeneousVector<BaseClass>::~HeterogeneousVector() {
  if (mOffsets == nullptr && mBytes == nullptr) {
    return;
  }
  destroy(true);
}

template <typename BaseClass>
void HeterogeneousVector<BaseClass>::destroy(bool isDestruct) {
  if (isDestruct) {
    for (size_t index = 0; index < mNextIndex; ++index) {
      reinterpret_cast<BaseClass*>(static_cast<unsigned char*>(mBytes) +
                                   mOffsets[index])
          ->~BaseClass();
    }
  }
  mAllocator.deallocateBytes(mOffsets, (mCapacity + 1) * sizeof(OffsetType));
  mAllocator.deallocateBytes(mBytes, mCapacity * mElementAverageSize);
}

template <typename BaseClass>
size_t HeterogeneousVector<BaseClass>::calculateRequiredCapacity(
    size_t addedElementSize) {
  size_t newCapacity = mCapacity;
  if (newCapacity <= mNextIndex) {
    newCapacity = newCapacity * 2 + 1;
  }
  auto offset = mCapacity == 0 ? 0 : mOffsets[mNextIndex];
  while (newCapacity * mElementAverageSize < offset + addedElementSize) {
    newCapacity *= 2;
  }
  return newCapacity;
}

template <typename BaseClass>
void HeterogeneousVector<BaseClass>::resize(size_t capacity) {
  if (capacity < mCapacity) {
    // TODO error
    return;
  }
  if (capacity == mCapacity) {
    return;
  }
  auto offsets = static_cast<OffsetType*>(
      mAllocator.allocateBytes((capacity + 1) * sizeof(OffsetType)));
  auto bytes = static_cast<byte*>(
      mAllocator.allocateBytes(capacity * mElementAverageSize));
  if (mBytes != nullptr && mOffsets != nullptr) {
    std::memcpy(offsets, mOffsets, (mCapacity + 1) * sizeof(OffsetType));
    std::memcpy(bytes, mBytes, mCapacity * mElementAverageSize);
    std::memset(mOffsets, 0, (mCapacity + 1) * sizeof(OffsetType));
    std::memset(mBytes, 0, (mCapacity)*mElementAverageSize);
  }
  std::memset(offsets + mCapacity, 0,
              (capacity + 1 - mCapacity) * sizeof(OffsetType));
  std::memset(static_cast<char*>(bytes) + mCapacity * mElementAverageSize, 0,
              (capacity - mCapacity) * mElementAverageSize);
  destroy(false);
  mOffsets = offsets;
  mBytes = bytes;
  mCapacity = capacity;
}

template <typename BaseClass>
size_t HeterogeneousVector<BaseClass>::size() const {
  return mNextIndex;
}

template <typename BaseClass>
template <typename Type>
Type& HeterogeneousVector<BaseClass>::get(size_t index) {
  return *reinterpret_cast<Type*>(static_cast<unsigned char*>(mBytes) +
                                  mOffsets[index]);
}

template <typename BaseClass>
template <typename Type>
const Type& HeterogeneousVector<BaseClass>::get(size_t index) const {
  return *reinterpret_cast<Type*>(static_cast<unsigned char*>(mBytes) +
                                  mOffsets[index]);
}

template <typename BaseClass>
template <typename Type>
bool HeterogeneousVector<BaseClass>::isAbleToContain() const {
  auto tmp = mCapacity > mNextIndex && mCapacity * mElementAverageSize >=
                                           mOffsets[mNextIndex] + sizeof(Type);
  return tmp;
}

template <typename BaseClass>
size_t HeterogeneousVector<BaseClass>::capacity() const {
  return mCapacity;
}

template <typename BaseClass>
size_t HeterogeneousVector<BaseClass>::elementAverageSize() const {
  return mElementAverageSize;
}
} // namespace athena::utils

#endif // ATHENA_HETEROGENEOUSVECTOR_H
