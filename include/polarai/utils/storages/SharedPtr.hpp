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

#include <polarai/utils/Memory.hpp>

#include <atomic>

namespace polarai::utils {
template <typename T> class SharedPtr {
public:
  using RefCntT = std::atomic_size_t;

  SharedPtr() = default;

  SharedPtr(T* ptr) : mPtr(ptr) {
    mRefCount = static_cast<RefCntT*>(allocate(sizeof(mPtr), alignof(RefCntT)));
    new (mRefCount) RefCntT{1};
  }

  template <typename U> SharedPtr(const SharedPtr<U>& other) {
    mRefCount = other.mRefCount;
    mPtr = static_cast<T*>(other.mPtr);
    mSameAllocation = other.mSameAllocation;

    (*mRefCount)++;
  }

  SharedPtr(const SharedPtr<T>& other) {
    mRefCount = other.mRefCount;
    mPtr = other.mPtr;
    mSameAllocation = other.mSameAllocation;

    (*mRefCount)++;
  }

  template <typename U> SharedPtr(SharedPtr<U>&& other) {
    mRefCount = other.mRefCount;
    mPtr = static_cast<U*>(other.mPtr);
    mSameAllocation = other.mSameAllocation;

    other.mRefCount = nullptr;
    other.mRefCount = nullptr;
  }

  SharedPtr(SharedPtr<T>&& other) {
    mRefCount = other.mRefCount;
    mPtr = other.mPtr;
    mSameAllocation = other.mSameAllocation;

    other.mRefCount = nullptr;
    other.mRefCount = nullptr;
  }

  template <typename U> SharedPtr<T>& operator=(const SharedPtr<U>& other) {
    mRefCount = other.mRefCount;
    mPtr = static_cast<T*>(other.mPtr);
    mSameAllocation = other.mSameAllocation;

    (*mRefCount)++;

    return *this;
  }

  SharedPtr<T>& operator=(const SharedPtr<T>& other) {
    mRefCount = other.mRefCount;
    mPtr = other.mPtr;
    mSameAllocation = other.mSameAllocation;

    (*mRefCount)++;

    return *this;
  }

  template <typename U> SharedPtr& operator=(SharedPtr<U>&& other) {
    mRefCount = other.mRefCount;
    mPtr = static_cast<U*>(other.mPtr);
    mSameAllocation = other.mSameAllocation;

    other.mRefCount = nullptr;
    other.mRefCount = nullptr;

    return *this;
  }

  SharedPtr& operator=(SharedPtr<T>&& other) {
    mRefCount = other.mRefCount;
    mPtr = other.mPtr;
    mSameAllocation = other.mSameAllocation;

    other.mRefCount = nullptr;
    other.mRefCount = nullptr;

    return *this;
  }

  ~SharedPtr() {
    if (!mRefCount) {
      return;
    }

    (*mRefCount)--;

    if (*mRefCount == 0) {
      // FIXME somehow we need to obtain size and alignment, otherwise mimalloc
      //   produces asserts.
      if (mSameAllocation) {
        deallocate(mRefCount, 0);
      } else {
        deallocate(mRefCount, sizeof(RefCntT), alignof(RefCntT));
        deallocate(const_cast<std::remove_cv_t<T>*>(mPtr), mSize, mAlignment);
      }
    }
  }

  T* get() { return mPtr; }

  size_t use_count() { return *mRefCount; }

  T* operator->() { return mPtr; }

  const T* operator->() const { return mPtr; }

  T& operator*() { return *mPtr; }

  const T& operator*() const { return *mPtr; }

  operator bool() { return mPtr != nullptr; }

private:
  SharedPtr(bool sameAlloc, RefCntT* cnt, T* ptr)
      : mRefCount(cnt), mPtr(ptr), mSameAllocation(sameAlloc) {
    (*mRefCount)++;
  }

  template <typename U> friend class SharedPtr;
  template <typename U> friend class WeakPtr;
  template <typename U, typename... Args>
  friend SharedPtr<U> makeShared(Args&&... args);

  // TODO replace atomic with own implementation?
  RefCntT* mRefCount = nullptr;
  T* mPtr = nullptr;
  bool mSameAllocation = false;
  size_t mSize = 0;
  size_t mAlignment = alignof(std::max_align_t);
};

// TODO remove. This is for compatibility with existing code only.
template <typename T, typename... Args>
SharedPtr<T> makeShared(Args&&... args) {
  void* mem = allocate(sizeof(T) + sizeof(typename SharedPtr<T>::RefCntT),
                       alignof(T), sizeof(typename SharedPtr<T>::RefCntT));
  new (mem) typename SharedPtr<T>::RefCntT;
  auto* refCount = static_cast<typename SharedPtr<T>::RefCntT*>(mem);
  auto* ptr = reinterpret_cast<T*>(static_cast<char*>(mem) +
                                   sizeof(typename SharedPtr<T>::RefCntT));

  new (ptr) T(std::forward<Args&&>(args)...);
  (*refCount) = 0;
  return SharedPtr<T>(true, refCount, ptr);
}
} // namespace polarai::utils
