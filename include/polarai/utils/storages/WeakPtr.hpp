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
#include <polarai/utils/storages/SharedPtr.hpp>

#include <atomic>

namespace polarai::utils {
template <typename T> class WeakPtr {
public:
  using RefCntT = std::atomic_size_t;

  WeakPtr() = default;

  template <typename U> WeakPtr(const SharedPtr<U>& ptr) {
    mRefCount = ptr.mRefCount;
    mPtr = ptr.mPtr;
    mSameAllocation = ptr.mSameAllocation;
  }

  template <typename U> WeakPtr(const WeakPtr<U>& ptr) {
    mRefCount = ptr.mRefCount;
    mPtr = ptr.mPtr;
    mSameAllocation = ptr.mSameAllocation;
  }

  WeakPtr(const WeakPtr<T>& other) {
    mRefCount = other.mRefCount;
    mPtr = other.mPtr;
    mSameAllocation = other.mSameAllocation;
  }

  template <typename U> WeakPtr(WeakPtr<U>&& other) {
    mRefCount = other.mRefCount;
    mPtr = static_cast<U*>(other.mPtr);
    mSameAllocation = other.mSameAllocation;

    other.mRefCount = nullptr;
    other.mRefCount = nullptr;
  }

  WeakPtr(WeakPtr<T>&& other) {
    mRefCount = other.mRefCount;
    mPtr = other.mPtr;
    mSameAllocation = other.mSameAllocation;

    other.mRefCount = nullptr;
    other.mRefCount = nullptr;
  }

  template <typename U> WeakPtr<T>& operator=(const WeakPtr<U>& other) {
    mRefCount = other.mRefCount;
    mPtr = static_cast<T*>(other.mPtr);
    mSameAllocation = other.mSameAllocation;

    return *this;
  }

  WeakPtr<T>& operator=(const WeakPtr<T>& other) {
    mRefCount = other.mRefCount;
    mPtr = other.mPtr;
    mSameAllocation = other.mSameAllocation;

    return *this;
  }

  template <typename U> WeakPtr& operator=(WeakPtr<U>&& other) {
    mRefCount = other.mRefCount;
    mPtr = static_cast<U*>(other.mPtr);
    mSameAllocation = other.mSameAllocation;

    other.mRefCount = nullptr;
    other.mRefCount = nullptr;

    return *this;
  }

  WeakPtr& operator=(WeakPtr<T>&& other) {
    mRefCount = other.mRefCount;
    mPtr = other.mPtr;
    mSameAllocation = other.mSameAllocation;

    other.mRefCount = nullptr;
    other.mRefCount = nullptr;

    return *this;
  }

  ~WeakPtr() = default;

  SharedPtr<T> lock() const noexcept {
    return SharedPtr<T>(mSameAllocation, mRefCount, mPtr);
  }

  size_t use_count() const noexcept { return *mRefCount; }

private:
  RefCntT* mRefCount = nullptr;
  T* mPtr = nullptr;
  bool mSameAllocation = false;
};

} // namespace polarai::utils
