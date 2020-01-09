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

#include "../../../src/backend/llvm/runtime/driver/RuntimeDriver.h"

#include <gtest/gtest.h>
#include <string>

namespace athena::backend::llvm {

TEST(RuntimeDriverTest, TestCreation) {
  RuntimeDriver driver;
  driver.load();
  ASSERT_TRUE(driver.isLoaded());
}

TEST(RuntimeDriverTest, TestFunctionLoad) {
  RuntimeDriver driver;
  driver.load();
  ASSERT_TRUE(driver.hasFeature("float"));
  ASSERT_TRUE(driver.hasBuiltin("add", "float"));
  ASSERT_FALSE(driver.hasBuiltin("foo", "float"));
}
} // namespace athena::backend::llvm
