/*
 * Copyright (c) 2019 Athena. All rights reserved.
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

#include "../../testutils/TestAllocator.h"

#include <athena/core/Context.h>
#include <athena/core/TensorShape.h>
#include <athena/core/inner/Tensor.h>
#include <athena/loaders/MemoryLoader/MemoryLoader.h>

#include <gtest/gtest.h>

TEST(LoadersTest, MemoryLoaderSingleLoad) {
  athena::core::Context context;
  // Arrange
  athena::core::TensorShape shape({1});
  athena::core::inner::Tensor tensor(athena::core::DataType::FLOAT, shape,
                                     context);

  TestAllocator allocator;
  allocator.allocate(tensor);

  float data[1];
  data[0] = 1;

  athena::loaders::MemoryLoader loader(&data, sizeof(float));

  // Act
  loader.load(&allocator, &tensor);

  // Assert
  float* res = reinterpret_cast<float*>(allocator.get(tensor));
  ASSERT_NE(res, nullptr);
  EXPECT_FLOAT_EQ(res[0], 1.0);
}

TEST(LoadersTest, MemoryLoaderComplexLoad) {
  athena::core::Context context;
  // Arrange
  athena::core::TensorShape shape({2, 3});
  athena::core::inner::Tensor tensor(athena::core::DataType::FLOAT, shape,
                                     context);

  TestAllocator allocator;
  allocator.allocate(tensor);

  float data[6];
  data[0] = 1;
  data[1] = 2;
  data[2] = 3;
  data[3] = 4;
  data[4] = 5;
  data[5] = 6;

  athena::loaders::MemoryLoader loader(&data, 6 * sizeof(float));

  // Act
  loader.load(&allocator, &tensor);

  // Assert
  float* res = reinterpret_cast<float*>(allocator.get(tensor));
  ASSERT_NE(res, nullptr);
  for (int i = 0; i < 6; i++) {
    EXPECT_FLOAT_EQ(res[i], (float)(i + 1));
  }
}

TEST(LoadersTest, MemoryLoaderCCreateName) {
  // Arrange
  athena::loaders::MemoryLoader loader(nullptr, 0);

  // Act
  auto name = loader.getCreateCName();

  // Assert
  EXPECT_EQ(name, "CreateMemoryLoader");
}

TEST(LoadersTest, MemoryLoaderCLoadName) {
  // Arrange
  athena::loaders::MemoryLoader loader(nullptr, 0);

  // Act
  auto name = loader.getLoadCName();

  // Assert
  EXPECT_EQ(name, "MemoryLoaderLoad");
}

TEST(LoadersTest, MemoryLoaderCCreateBind) {
  // Arrange
  auto pLoader = reinterpret_cast<athena::loaders::MemoryLoader*>(
      CreateMemoryLoader(nullptr, 0));

  // Assert
  ASSERT_NE(pLoader, nullptr);
}

TEST(LoadersTest, MemoryLoaderCLoadBind) {
  athena::core::Context context;
  // Arrange
  athena::core::TensorShape shape({1});
  athena::core::inner::Tensor tensor(athena::core::DataType::FLOAT, shape,
                                     context);

  TestAllocator allocator;
  allocator.allocate(tensor);

  float data[1];
  data[0] = 1;

  auto pLoader = reinterpret_cast<athena::loaders::MemoryLoader*>(
      CreateMemoryLoader(data, sizeof(float)));

  // Act
  MemoryLoaderLoad(pLoader, &allocator, &tensor);

  // Assert
  float* res = reinterpret_cast<float*>(allocator.get(tensor));
  ASSERT_NE(res, nullptr);
  EXPECT_FLOAT_EQ(res[0], 1.0);
}