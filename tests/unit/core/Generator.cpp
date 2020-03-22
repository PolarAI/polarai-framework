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

#include <athena/core/Context.h>
#include <athena/core/Generator.h>
#include <athena/core/inner/Tensor.h>

#include <any>
#include <gtest/gtest.h>
#include <vector>

using namespace athena::core;

struct DummyState {
  bool flag = false;
};

TEST(GeneratorTest, AddSimpleFunctor) {
  // Arrange
  Context ctx;
  auto state = std::make_shared<DummyState>();
  Generator generator(ctx, state);

  auto f = [&](Context& ctx, std::string_view graphName, size_t nodeId,
               size_t clusterId, const std::vector<inner::Tensor>& args,
               const std::any& opts) {};

  // Act
  generator.registerFunctor("test", f);

  // Assert
  ASSERT_TRUE(generator.hasFunctor("test"));
}

TEST(GeneratorTest, RemoveFunctor) {
  // Arrange
  Context ctx;
  auto state = std::make_shared<DummyState>();
  Generator generator(ctx, state);

  auto f = [&](Context& ctx, std::string_view graphName, size_t nodeId,
               size_t clusterId, const std::vector<inner::Tensor>& args,
               const std::any& opts) {};

  // Act
  generator.registerFunctor("test", f);
  EXPECT_TRUE(generator.hasFunctor("test"));
  generator.unregisterFunctor("test");

  // Assert
  EXPECT_FALSE(generator.hasFunctor("test"));
}

TEST(GeneratorTest, BehavesCorrectly) {
  // Arrange
  Context ctx;
  auto state = std::make_shared<DummyState>();
  Generator generator(ctx, state);

  generator.setGenerationPoint("TestGraph", 1, 2);

  auto f = [state](Context& ctx, std::string_view graphName, size_t nodeId,
                   size_t clusterId, const std::vector<inner::Tensor>& args,
                   const std::any& opts) {
    EXPECT_EQ(graphName, "TestGraph");
    EXPECT_EQ(nodeId, 1);
    EXPECT_EQ(clusterId, 2);
    state->flag = true;
  };
  generator.registerFunctor("test", f);

  // Act
  generator.generate("test", {});

  // Assert
  ASSERT_TRUE(state->flag);
}
