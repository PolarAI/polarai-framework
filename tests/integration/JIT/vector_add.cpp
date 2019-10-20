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
#include <athena/core/GradientDescent.h>
#include <athena/core/Graph.h>
#include <athena/core/InputNode.h>
#include <athena/core/LossNode.h>
#include <athena/core/Node.h>
#include <athena/core/inner/Tensor.h>
#include <athena/loaders/MemoryLoader/MemoryLoader.h>
#include <athena/ops/AddOperation.h>
#include <athena/ops/MSELossFunction.h>

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
    float cData[] = {0, 0, 0};

    MemoryLoader aLoader(aData, 3 * sizeof(float));
    MemoryLoader bLoader(bData, 3 * sizeof(float));
    MemoryLoader cLoader(cData, 3 * sizeof(float));

    Graph graph;
    graph.setUpOptimizer<GradientDescent>(/*learningRate*/ 0.01);
    InputNode aInp(shape, DataType::FLOAT, aLoader, false, "a");
    InputNode bInp(shape, DataType::FLOAT, bLoader, false, "b");
    graph.addNode(aInp);
    graph.addNode(bInp);

    AddOperation addOp;
    Node add(addOp, "vector_add_1");
    graph.addNode(add);
    add.after(aInp, 1);
    add.after(bInp, 2);

    OutputNode outputNode(DataType::FLOAT, "out");
    graph.addNode(outputNode);
    outputNode.after(add, 1);

    MSELossFunction lossFunction;
    InputNode cInp(shape, DataType::FLOAT, cLoader, true, "c");
    graph.addNode(cInp);
    LossNode lossNode(lossFunction, Criterion::MIN, "mse_loss");
    graph.addNode(lossNode);
    lossNode.after(add, 1);
    lossNode.after(cInp, 2);

    LLVMExecutor executor;
    std::unique_ptr<Allocator> trivialAllocator =
        std::make_unique<LLVMTrivialAllocator>();
    executor.setAllocator(trivialAllocator);
    executor.setGraph(graph);

    // Act
    executor.evaluate();

    // Assert
    auto accessor = outputNode.getAccessor<float>(*executor.getAllocator());

    EXPECT_FLOAT_EQ(*accessor[0], 5.0);
    EXPECT_FLOAT_EQ(*accessor[1], 7.0);
    EXPECT_FLOAT_EQ(*accessor[2], 9.0);
}