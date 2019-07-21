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

#include <athena/core/FatalError.h>
#include <athena/core/ShapeView.h>
#include <athena/core/TensorShape.h>

#include <gtest/gtest.h>
#include <numeric>

namespace athena::core {
TEST(TensorViewTest, CreationEmpty) {
    TensorShape shape{};
    ShapeView view = shape.getShapeView();
    ASSERT_EQ(view.dimensions(), 0);
    ASSERT_EQ(view.getTotalSize(), 0);
}
TEST(TensorViewTest, CreationFromShape) {
    std::vector<size_t> expected{3, 2, 3, 20};
    size_t product = std::accumulate<std::vector<size_t>::iterator, size_t>(
        expected.begin(), expected.end(), 1, std::multiplies<>());
    TensorShape shape(expected);
    ShapeView view = shape.getShapeView();
    ASSERT_EQ(view.dimensions(), expected.size());
    ASSERT_EQ(view.getTotalSize(), product);
}
TEST(TensorViewTest, CheckFieldsAndFunctions) {
    std::vector<size_t> expected{12, 12, 3, 2, 3, 20};
    size_t product = std::accumulate<std::vector<size_t>::iterator, size_t>(
        expected.begin(), expected.end(), 1, std::multiplies<>());
    TensorShape shape(expected);
    ShapeView view = shape.getShapeView();
    ASSERT_EQ(view.dimensions(), expected.size());
    ASSERT_EQ(view.getTotalSize(), product);
    size_t index = 0;
    for (auto elem : view) {
        ASSERT_EQ(elem, expected[index]);
        ++index;
    }
    for (size_t index_sec = 0; index_sec < expected.size(); ++index_sec) {
        ASSERT_EQ(view[index_sec], expected[index_sec]);
    }
    ShapeView subview = view.getSubShapeView();
    ASSERT_EQ(subview.dimensions(), expected.size() - 1);
    ASSERT_EQ(subview.getTotalSize(), product / expected[0]);
    index = 1;
    for (auto elem : subview) {
        ASSERT_EQ(elem, expected[index]);
        ++index;
    }
}
}  // namespace athena::core
