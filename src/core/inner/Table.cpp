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

#include <athena/core/inner/Table.h>

#include <vector>
namespace athena::core::inner {
Table<AllocationRecord>::Table() : mLastId(1) {
  mRegisteredContents.emplace_back(mNullRecord);
}

AllocationRecord& Table<AllocationRecord>::operator[](size_t index) {
  return mRegisteredContents[index];
}

size_t Table<AllocationRecord>::size() { return mRegisteredContents.size(); }

void Table<AllocationRecord>::clear() {
  mRegisteredContents.clear();
  mRegisteredContents.emplace_back(mNullRecord);
  mLastId = 1;
}
} // namespace athena::core::inner