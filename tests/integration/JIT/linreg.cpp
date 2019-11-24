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
#include <athena/core/inner/InnerFunctions.h>
#include <athena/core/inner/Tensor.h>
#include <athena/loaders/MemoryLoader/MemoryLoader.h>
#include <athena/model/DotModel.h>
#include <athena/ops/GEMMOperation.h>
#include <athena/ops/MSELossFunction.h>

#include <gtest/gtest.h>

using namespace athena::core;
using namespace athena::ops;
using namespace athena::backend::llvm;
using namespace athena::loaders;

TEST(JIT, LinReg) {
    // Arrange
    TensorShape shape({1, 9});
    TensorShape shape2({9, 1});
    TensorShape shapeScalar({1});

    float input[] = {10, 20, 20, 20, 20, 20, 20, 70, 50};
    float weights[] = {3, 3, 3, 3, 3, 3, 3, 3, 3};
    float target[] = {1};

    MemoryLoader inputLoader(input, 9 * sizeof(float));
    MemoryLoader weightsLoader(weights, 9 * sizeof(float));
    MemoryLoader targetLoader(target, 1 * sizeof(float));

    Context context;
    Graph graph(context);
    graph.setUpOptimizer<GradientDescent>(/*learningRate*/ -0.0000001);
    InputNode inputInp(shape, DataType::FLOAT, inputLoader, context, false,
                       "a");
    InputNode weightsInp(shape2, DataType::FLOAT, weightsLoader, context, false,
                         "b");
    graph.addNode(inputInp);
    graph.addNode(weightsInp);

    OutputNode outputNodeDbg(DataType::FLOAT, context, "debugger");
    graph.addNode(outputNodeDbg);
    outputNodeDbg.after(weightsInp, 1);

    GEMMOperation gemmOp(false, false);
    Node gemm(gemmOp, context, "gemm_1");
    graph.addNode(gemm);
    gemm.after(inputInp, 1);
    gemm.after(weightsInp, 2);

    OutputNode outputNode(DataType::FLOAT, context, "out");
    graph.addNode(outputNode);
    outputNode.after(gemm, 2);

    MSELossFunction lossFunction;
    InputNode cInp(shapeScalar, DataType::FLOAT, targetLoader, context, true,
                   "c");
    graph.addNode(cInp);
    LossNode lossNode(lossFunction, Criterion::MIN, context, "mse_loss");
    graph.addNode(lossNode);
    lossNode.after(gemm, 1);
    lossNode.after(cInp, 2);

    OutputNode lossOut(DataType::FLOAT, context, "lossOut");
    graph.addNode(lossOut);
    lossOut.after(lossNode, 1);

    LLVMExecutor executor;
    std::unique_ptr<Allocator> trivialAllocator =
        std::make_unique<LLVMTrivialAllocator>();
    executor.setAllocator(trivialAllocator);
    executor.setGraph(graph);

    // Act
    executor.evaluate();
    executor.optimizeGraph();

    // Assert
    auto accessor = outputNode.getAccessor<float>(*executor.getAllocator());

    EXPECT_FLOAT_EQ(*accessor[0][0], 750.0f);

    auto accessorWeights =
        outputNodeDbg.getAccessor<float>(*executor.getAllocator());

    EXPECT_FLOAT_EQ(*accessorWeights[0][0], 1.873498);
    EXPECT_FLOAT_EQ(*accessorWeights[0][1], 0.746996);
    EXPECT_FLOAT_EQ(*accessorWeights[0][2], 0.746996);
    EXPECT_FLOAT_EQ(*accessorWeights[0][3], 0.746996);
    EXPECT_FLOAT_EQ(*accessorWeights[0][4], 0.746996);
    EXPECT_FLOAT_EQ(*accessorWeights[0][5], 0.746996);
    EXPECT_FLOAT_EQ(*accessorWeights[0][6], 0.746996);
    EXPECT_FLOAT_EQ(*accessorWeights[0][7], -4.8855138);
    EXPECT_FLOAT_EQ(*accessorWeights[0][8], -2.63251);
}
