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

#include <athena/backend/llvm/LLVMExecutor.h>
#include <athena/backend/llvm/LLVMTrivialAllocator.h>
#include <athena/core/Graph.h>
#include <athena/core/InputNode.h>
#include <athena/core/Node.h>
#include <athena/core/inner/Tensor.h>
#include <athena/loaders/MemoryLoader/MemoryLoader.h>
#include <athena/ops/AddOperation.h>

#include <gtest/gtest.h>

using namespace athena::core;
using namespace athena::ops;
using namespace athena::backend::llvm;
using namespace athena::loaders;

TEST(JIT, SimpleVectorAdd) {
    // Arrange
    TensorShape shape({3});

    float aData[] = {1, 2, 3};
    float bData[] = {4, 5, 6};

    MemoryLoader aLoader(aData, 3 * sizeof(float));
    MemoryLoader bLoader(bData, 3 * sizeof(float));

    Graph graph;
    InputNode aInp(shape, DataType::FLOAT, aLoader);
    InputNode bInp(shape, DataType::FLOAT, bLoader);
    graph.addNode(aInp);
    graph.addNode(bInp);

    AddOperation addOp;
    Node add(shape, DataType::FLOAT, addOp, "vector_add_1");
    graph.addNode(add);
    add.after(aInp, 1);
    add.after(bInp, 2);

    LLVMExecutor executor;
    std::unique_ptr<Allocator> trivialAllocator =
        std::make_unique<LLVMTrivialAllocator>();
    executor.setAllocator(trivialAllocator);
    executor.prepare(graph);

    // Act
    executor.execute();

    // Assert

    auto pRes = (float*)executor.getAllocator()->getFastPointer(
        inner::getTensorFromNode(add));

    EXPECT_FLOAT_EQ(pRes[0], 5.0);
    EXPECT_FLOAT_EQ(pRes[1], 7.0);
    EXPECT_FLOAT_EQ(pRes[2], 9.0);
}