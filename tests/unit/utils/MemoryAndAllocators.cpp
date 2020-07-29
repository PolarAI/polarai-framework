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

#include <polarai/utils/Memory.hpp>
#include <polarai/utils/allocator/Allocator.hpp>

#include <gtest/gtest.h>

using namespace polarai::utils;

TEST(MemoryAndAllocators, FreestandingFunctions) {
  void* p1 = allocate(10);
  ASSERT_NE(p1, nullptr);
  deallocate(p1, 10);

  void* p2 = allocate(10, 8);
  ASSERT_NE(p2, nullptr);
  deallocate(p2, 10, 8);

  void* p3 = allocate(24, 8, 8);
  ASSERT_NE(p3, nullptr);
  deallocate(p3, 10, 8);
}

TEST(MemoryAndAllocators, StdContainer) {
  std::vector<int, Allocator<int>> vec;
  vec.push_back(42);
  ASSERT_EQ(vec.front(), 42);

  std::vector<double, Allocator<double>> vec2;
  vec2.push_back(42.0);
  ASSERT_FLOAT_EQ(vec2.front(), 42.0);

  using StrT = std::basic_string<char, std::char_traits<char>, Allocator<char>>;

  std::vector<StrT, Allocator<StrT>> strvec;
  strvec.push_back("abcd");
  ASSERT_EQ(strvec.front(), "abcd");

}
