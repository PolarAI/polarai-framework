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

#include "../../../../src/backend/llvm/allocators/TrivialAllocator.h"

#include <gtest/gtest.h>

using namespace athena::backend::llvm;

TEST(LLVMBackend, TrivialAllocatorSimple) {
  TrivialAllocator allocator;

  MemoryRecord record{1, 30};
  allocator.allocate(record);
  auto ptr = allocator.getPtr(record);

  ASSERT_NE(ptr, nullptr);
}
