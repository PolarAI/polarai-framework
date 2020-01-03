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

#include <athena/core/InputNode.h>

#include <gtest/gtest.h>
#include <string>

namespace athena::core {
TEST(InputNode, Create) {
  Context context;
  TensorShape shape{2, 3, 4};
  DataType dataType = DataType::DOUBLE;
  std::string name = "Dummy";
  DummyLoader loader;
  InputNode n(shape, dataType, loader, context, false, name);
  ASSERT_EQ(n.getDataType(), dataType);
  ASSERT_EQ(n.getShapeView(), shape);
  ASSERT_EQ(n.getType(), NodeType::INPUT);
  ASSERT_EQ(n.getName(), name);
  ASSERT_EQ(&n.getLoader(), &loader);
}
TEST(InputNode, CopyConstructor) {
  Context context;
  TensorShape shape{2, 3, 4};
  DataType dataType = DataType::DOUBLE;
  std::string name = "Dummy";
  DummyLoader loader;
  InputNode n(shape, dataType, loader, context, false, name);
  InputNode nSecond(n);
  ASSERT_EQ(inner::getNodeTable(context).size(), 3);
  ASSERT_EQ(n.getShapeView(), nSecond.getShapeView());
  ASSERT_EQ(n.getType(), nSecond.getType());
  ASSERT_EQ(n.getDataType(), nSecond.getDataType());
  ASSERT_EQ(n.getName(), nSecond.getName());
  ASSERT_EQ(n.name(), nSecond.name());
  ASSERT_EQ(&n.getLoader(), &loader);
}
} // namespace athena::core
