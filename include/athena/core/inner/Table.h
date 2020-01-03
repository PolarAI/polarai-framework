/*
 * Copyright (c) 2018 Athena. All rights reserved.
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

#ifndef ATHENA_TABLE_H
#define ATHENA_TABLE_H

#include <athena/core/core_export.h>
#include <athena/core/inner/NullRecord.h>

#include <vector>

namespace athena::core::inner {
constexpr size_t kKUndefinedIndex = 0;

template <typename Content> class ATH_CORE_EXPORT Table {
private:
  static inline Content const mNullRecord = NullRecord<Content>::value;
  std::vector<Content> mRegisteredContents;

public:
  Table();
  Table(const Table&) = delete;
  Table(Table&& rhs) = delete;
  ~Table() = default;

  Table& operator=(const Table& rhs) = delete;
  Table& operator=(Table&& rhs) = delete;
  Content& operator[](size_t index);

  template <typename... Args> size_t registerRecord(Args&&... args);
  size_t size();
  void clear();
};

template <typename Content> ATH_CORE_EXPORT Table<Content>::Table() {
  mRegisteredContents.emplace_back(mNullRecord);
}

template <typename Content>
ATH_CORE_EXPORT Content& Table<Content>::operator[](size_t index) {
  return mRegisteredContents[index];
}

template <typename Content>
template <typename... Args>
ATH_CORE_EXPORT size_t Table<Content>::registerRecord(Args&&... args) {
  mRegisteredContents.emplace_back(std::forward<Args>(args)...);
  return mRegisteredContents.size() - 1;
}

template <typename Content> ATH_CORE_EXPORT size_t Table<Content>::size() {
  return mRegisteredContents.size();
}

template <typename Content> ATH_CORE_EXPORT void Table<Content>::clear() {
  mRegisteredContents.clear();
  mRegisteredContents.emplace_back(mNullRecord);
}

template <> class ATH_CORE_EXPORT Table<AllocationRecord> {
private:
  size_t mLastId;
  static inline const AllocationRecord mNullRecord =
      NullRecord<AllocationRecord>::value;
  std::vector<AllocationRecord> mRegisteredContents;

public:
  Table();
  Table(const Table&) = delete;
  Table(Table&& rhs) = delete;
  ~Table() = default;

  Table& operator=(const Table& rhs) = delete;
  Table& operator=(Table&& rhs) = delete;
  AllocationRecord& operator[](size_t index);

  template <typename... Args> size_t registerRecord(Args&&... args);
  size_t size();
  void clear();
};

template <typename... Args>
ATH_CORE_EXPORT size_t Table<AllocationRecord>::registerRecord(Args&&... args) {
  mRegisteredContents.emplace_back(std::forward<Args>(args)...);
  size_t lastIdCopy = mLastId;
  mLastId += mRegisteredContents.back().getSize();
  return lastIdCopy;
}
} // namespace athena::core::inner

#endif // ATHENA_TABLE_H
