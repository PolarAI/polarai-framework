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

#include "../../../src/backend/llvm/runtime/legacy_driver/runtime-driver.h"

#include <gtest/gtest.h>
#include <llvm/Support/TargetSelect.h>
#include <string>

static const std::string kPathToRuntimeCPUName = "PATH_TO_RUNTIME_CPU";

namespace athena::backend::llvm {

class RuntimeDriverTest : public ::testing::Test {
protected:
  std::string mPathToRuntimeCPU;
  std::unique_ptr<::llvm::LLVMContext> mContext =
      std::make_unique<::llvm::LLVMContext>();
  LegacyRuntimeDriver mDriver;

  void SetUp() override {
    ::llvm::InitializeNativeTarget();
    ::llvm::InitializeNativeTargetAsmParser();
    ::llvm::InitializeNativeTargetAsmPrinter();
    mPathToRuntimeCPU = ::getenv(kPathToRuntimeCPUName.data());
  }

public:
  RuntimeDriverTest() : mDriver(*mContext) {}
};

TEST_F(RuntimeDriverTest, TestCreation) {
  mDriver.reload(mPathToRuntimeCPU);
  ASSERT_TRUE(mDriver.isLoaded());
}

TEST_F(RuntimeDriverTest, TestFunctionLoad) {
  mDriver.load(mPathToRuntimeCPU);
  auto& modules = mDriver.getModules();
  ASSERT_GT(modules.size(), 0);

  ASSERT_NE(modules[0]->getFunction("athn_add_f"), nullptr);
}
} // namespace athena::backend::llvm
