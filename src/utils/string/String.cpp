/*
 * Copyright (c) 2019 Athena. All rights reserved.
 * https://getathena.ml
 *
 * Licensed under MIT license.
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS, WITHOUT
 * WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the
 * License for the specific language governing permissions and limitations under
 * the License.
 */

#include "athena/utils/string/String.h"
#include "athena/utils/error/FatalError.h"
#include <cstring>
#include <iostream>

namespace athena::utils {
String::String() : mSize(0), mData(nullptr), mAllocator(Allocator()) {}

String::String(const char* const string, Allocator allocator)
    : mSize(strlen(string)),
      mData(reinterpret_cast<const char*>(
          allocator.allocateBytes((mSize + 1) * sizeof(char)))),
      mAllocator(std::move(allocator)) {
//  std::cout << "Creating string: " << static_cast<const void*>(mData) <<
//  std::endl;
// std::cout << "Data: " << static_cast<const void*>(mData) << std::endl;
#ifdef DEBUG
  if (mData == nullptr) {
    FatalError(ATH_ASSERT, "Memory allocation for string ", this,
               " didn't perform.");
  }
#endif
  memcpy((void*)mData, string, (mSize + 1) * sizeof(char));
}

String::String(String&& rhs) noexcept
    : mSize(rhs.mSize), mData(rhs.mData),
      mAllocator(std::move(rhs.mAllocator)) {
  rhs.mSize = 0;
  rhs.mData = nullptr;
  //  std::cout << "Moving string: " << static_cast<const void*>(mData) <<
  //  std::endl;
}

String::~String() {
//  std::cout << "Deleting string: " << static_cast<const void*>(mData) <<
//  std::endl;
#ifdef DEBUG
  if (mData != nullptr && mSize != strlen(mData)) {
    //    FatalError(ATH_ASSERT, "Size of string ", this, " isn't equal to
    //    actual size.");
  }
#endif
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
} // namespace athena::utils
