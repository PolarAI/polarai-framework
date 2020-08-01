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

#include <polarai/utils/storages/SharedPtr.hpp>
#include <polarai/utils/storages/WeakPtr.hpp>

#include <gtest/gtest.h>

using namespace polarai::utils;

TEST(Pointers, SharedPtrSimple) {
  int* myInt = static_cast<int*>(allocate(sizeof(int), alignof(int)));
  SharedPtr<int> ptr{myInt};

  EXPECT_TRUE((bool)ptr);
  EXPECT_EQ(ptr.use_count(), 1);

  {
    SharedPtr<int> copy = ptr;
    EXPECT_EQ(ptr.use_count(), 2);
    EXPECT_EQ(copy.use_count(), 2);
  }

  EXPECT_EQ(ptr.use_count(), 1);
}

TEST(Pointers, MakeSharedSimple) {
  SharedPtr<int> ptr = makeShared<int>(10);

  ASSERT_NE(ptr.get(), nullptr);
  EXPECT_EQ(*ptr, 10);
}

TEST(Pointers, WeakPtrSimple) {
  SharedPtr<int> ptr = makeShared<int>(10);

  ASSERT_NE(ptr.get(), nullptr);
  EXPECT_EQ(*ptr, 10);
  EXPECT_EQ(ptr.use_count(), 1);

  WeakPtr<int> wptr = ptr;
  EXPECT_EQ(wptr.use_count(), 1);
  EXPECT_EQ(ptr.use_count(), 1);

  {
    SharedPtr<int> locked = wptr.lock();
    EXPECT_EQ(wptr.use_count(), 2);
    EXPECT_EQ(ptr.use_count(), 2);
    EXPECT_EQ(locked.use_count(), 2);
  EXPECT_EQ(*locked, 10);
  }

  EXPECT_EQ(wptr.use_count(), 1);
  EXPECT_EQ(ptr.use_count(), 1);
}
