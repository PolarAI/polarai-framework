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

#include <polarai/utils/error/FatalError.hpp>
#include <polarai/utils/string/String.hpp>

#include <cstring>
#include <iostream>

namespace polarai::utils {
String::String() : mSize(0), mAllocator(Allocator()), mData(nullptr) {}

String::String(const char* const string, Allocator allocator)
    : mSize(strlen(string)), mAllocator(std::move(allocator)),
      mData(reinterpret_cast<const char*>(
          mAllocator.allocateBytes((mSize + 1) * sizeof(char)))) {
#ifdef DEBUG
  if (mData == nullptr) {
    FatalError(ATH_ASSERT, "Memory allocation for string ", this,
               " didn't perform.");
  }
#endif
  memcpy((void*)mData, string, (mSize + 1) * sizeof(char));
}

String::String(const String& rhs)
    : mSize(rhs.mSize), mAllocator(rhs.mAllocator),
      mData(reinterpret_cast<const char*>(
          mAllocator.allocateBytes((mSize + 1) * sizeof(char)))) {
  memcpy((void*)mData, rhs.mData, (mSize + 1) * sizeof(char));
}

String::String(String&& rhs) noexcept
    : mSize(rhs.mSize), mData(rhs.mData),
      mAllocator(std::move(rhs.mAllocator)) {
  rhs.mSize = 0;
  rhs.mData = nullptr;
}

String::~String() {
  if (mData == nullptr) {
    return;
  }
  // TODO add define for safety mode with memory filling by zeros
  mAllocator.deallocateBytes(mData, (mSize + 1) * sizeof(char));
}

const char* String::getString() const {
#ifdef DEBUG
  if (mSize != strlen(mData)) {
    FatalError(ATH_ASSERT, "Size of string ", this,
               " isn't equal to actual size.");
  }
#endif
  return mData;
}

size_t String::getSize() const {
#ifdef DEBUG
  if (mSize != strlen(mData)) {
    FatalError(ATH_ASSERT, "Size of string ", this,
               " isn't equal to actual size.");
  }
#endif
  return mSize;
}
} // namespace polarai::utils
