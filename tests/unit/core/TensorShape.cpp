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

#include <athena/core/TensorShape.h>

#include <gtest/gtest.h>
#include <numeric>

namespace athena::core {
TEST(TensorShapeTest, CreationWithInitList) {
  TensorShape shape{3, 2, 3};
  std::vector<size_t> expected{3, 2, 3};
  size_t product = std::accumulate<std::vector<size_t>::iterator, size_t>(
      expected.begin(), expected.end(), 1, std::multiplies<size_t>());
  ASSERT_EQ(shape.getShape(), expected);
  ASSERT_EQ(shape.getTotalSize(), product);
}

TEST(TensorShapeTest, CreationEmpty) {
  TensorShape shape{};
  std::vector<size_t> expected{};
  ASSERT_EQ(shape.getShape(), expected);
  ASSERT_EQ(shape.getTotalSize(), 0);
}

TEST(TensorShapeTest, CreationWithVector) {
  std::vector<size_t> expected{3, 2, 3, 20};
  size_t product = std::accumulate<std::vector<size_t>::iterator, size_t>(
      expected.begin(), expected.end(), 1, std::multiplies<size_t>());
  TensorShape shape(expected);
  for (size_t index_sec = 0; index_sec < expected.size(); ++index_sec) {
    ASSERT_EQ(shape[index_sec], expected[index_sec]);
  }
  ASSERT_EQ(shape.getShape(), expected);
  ASSERT_EQ(shape.getTotalSize(), product);
}

TEST(TensorShapeTest, CreationWithCopyableIterator) {
  std::vector<int16_t> data{3, 2, 3, 20};
  std::vector<size_t> expected(data.begin(), data.end());
  int16_t product = std::accumulate<std::vector<int16_t>::iterator, int16_t>(
      data.begin(), data.end(), 1, std::multiplies<int16_t>());
  TensorShape shape(data.begin(), data.end());
  for (size_t index_sec = 0; index_sec < data.size(); ++index_sec) {
    ASSERT_EQ(shape[index_sec], data[index_sec]);
  }
  ASSERT_EQ(shape.getShape(), expected);
  ASSERT_EQ(shape.getTotalSize(), product);
}

TEST(TensorShapeTest, Dimension) {
  TensorShape shape{1, 2, 3, 4, 2};
  ASSERT_EQ(shape.dimensions(), 5);
}

TEST(TensorShapeTest, Dim) {
  std::vector<size_t> expected{10, 2, 3, 4, 2};
  TensorShape shape(expected);
  for (size_t index = 0; index < expected.size(); ++index) {
    ASSERT_EQ(shape.dim(index), expected[index]);
  }
}

TEST(TensorShapeTest, TotalSize) {
  std::vector<size_t> expected{10, 2, 3, 4, 2};
  TensorShape shape(expected);
  size_t product = std::accumulate<std::vector<size_t>::iterator, size_t>(
      expected.begin(), expected.end(), 1, std::multiplies<size_t>());
  ASSERT_EQ(shape.getTotalSize(), product);
}

TEST(TensorShapeTest, RangeBasedFor) {
  std::vector<size_t> expected{10, 2, 3, 4, 2};
  TensorShape shape(expected);
  size_t index = 0;
  for (size_t value : shape) {
    ASSERT_EQ(value, expected[index]);
    ++index;
  }
}

TEST(TensorShapeTest, SubShape) {
  TensorShape shape{3, 2, 3};
  std::vector<size_t> expected{3, 2, 3};
  TensorShape shapeSecond(
      std::vector<size_t>(expected.begin() + 1, expected.end()));
  ASSERT_EQ(shape.getSubShape(), shapeSecond);
}
} // namespace athena::core
