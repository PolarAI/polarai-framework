/*
 * Copyright (c) 2018 Athena. All rights reserved.
 * https://athenaframework.ml
 *
 * Licensed under MIT license.
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an “AS IS” BASIS, WITHOUT
 * WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the
 * License for the specific language governing permissions and limitations under
 * the License.
 */

#include <athena/core/Node.h>
#include <athena/core/inner/GlobalTables.h>

#include <gtest/gtest.h>
#include <string>

namespace athena::core {
TEST(Node, Create) {
    TensorShape shape{2, 3, 4};
    DataType dataType = DataType::DOUBLE;
    std::string operationName = "Dummy";
    OperationDummy op(operationName);
    Node n(shape, dataType, op);
    ASSERT_EQ(n.getDataType(), dataType);
    ASSERT_EQ(n.getShapeView(), shape);
    ASSERT_EQ(n.getType(), NodeType::DEFAULT);
    ASSERT_EQ(n.getName(), "");
    ASSERT_EQ(&n.getOperation(), &op);
}
TEST(Node, CopyConstructor) {
    inner::getNodeTable().clear();
    TensorShape shape{2, 3, 4};
    DataType dataType = DataType::DOUBLE;
    std::string operationName = "Dummy";
    OperationDummy op(operationName);
    Node n(shape, dataType, op);
    Node nSecond(n);
    ASSERT_EQ(inner::getNodeTable().size(), 3);
    ASSERT_EQ(n.getShapeView(), nSecond.getShapeView());
    ASSERT_EQ(n.getType(), nSecond.getType());
    ASSERT_EQ(n.getDataType(), nSecond.getDataType());
    ASSERT_EQ(n.getName(), nSecond.getName());
    ASSERT_EQ(n.name(), nSecond.name());
}
TEST(Node, CopyOperator) {
    inner::getNodeTable().clear();
    TensorShape shape{2, 3, 4};
    DataType dataType = DataType::DOUBLE;
    std::string operationName = "Dummy";
    OperationDummy op(operationName);
    Node n(shape, dataType, op);
    TensorShape shapeSecond{2, 3, 4};
    DataType dataTypeSecond = DataType::HALF;
    std::string operationNameSecond = "DummySecond";
    Node nSecond(shapeSecond, dataTypeSecond, op, operationNameSecond);
    nSecond = n;
    ASSERT_EQ(inner::getNodeTable().size(), 3);
    ASSERT_EQ(n.getShapeView(), nSecond.getShapeView());
    ASSERT_EQ(n.getType(), nSecond.getType());
    ASSERT_EQ(n.getDataType(), nSecond.getDataType());
    ASSERT_EQ(n.getName(), nSecond.getName());
    ASSERT_EQ(n.name(), nSecond.name());
}
TEST(Node, NodeSavesOperation) {
    inner::getNodeTable().clear();
    OperationDummy op("DummyOp");
    Node node({1}, DataType::DOUBLE, op);

    EXPECT_EQ(node.getOperation().getName(), "DummyOp");
}
}  // namespace athena::core