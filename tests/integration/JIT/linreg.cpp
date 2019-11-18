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
    TensorShape shapeScalar({1});

    float input[] = {10, 20, 20, 20, 20, 20, 20, 70, 50};
    float weights[] = {3, 3, 3, 3, 3, 3, 3, 3, 3};
    float target[] = {1};

    MemoryLoader inputLoader(input, 9 * sizeof(float));
    MemoryLoader weightsLoader(weights, 9 * sizeof(float));
    MemoryLoader targetLoader(target, 1 * sizeof(float));

    Context context;
    Graph graph(context);
    graph.setUpOptimizer<GradientDescent>(/*learningRate*/ 0.001);
    InputNode inputInp(shape, DataType::FLOAT, inputLoader, context, false, "a");
    InputNode weightsInp(shape, DataType::FLOAT, weightsLoader, context, false, "b");
    graph.addNode(inputInp);
    graph.addNode(weightsInp);

    OutputNode outputNodeDbg(DataType::FLOAT, context, "debugger");
    graph.addNode(outputNodeDbg);
    outputNodeDbg.after(weightsInp, 1);

    GEMMOperation gemmOp(false, true);
    Node gemm(gemmOp, context, "gemm_1");
    graph.addNode(gemm);
    gemm.after(inputInp, 1);
    gemm.after(weightsInp, 2);

    OutputNode outputNode(DataType::FLOAT, context, "out");
    graph.addNode(outputNode);
    outputNode.after(gemm, 1);

    MSELossFunction lossFunction;
    InputNode cInp(shapeScalar, DataType::FLOAT, targetLoader, context, true,
                   "c");
    graph.addNode(cInp);
    LossNode lossNode(lossFunction, Criterion::MIN, context, "mse_loss");
    graph.addNode(lossNode);
    lossNode.after(gemm, 1);
    lossNode.after(cInp, 2);

    LLVMExecutor executor;
    std::unique_ptr<Allocator> trivialAllocator =
        std::make_unique<LLVMTrivialAllocator>();
    executor.setAllocator(trivialAllocator);
    executor.setGraph(graph);

    athena::model::DotModel::exportGraph(graph, std::cerr);

    // Act
    executor.evaluate();
    executor.optimizeGraph();

    // Assert
    auto accessor = outputNode.getAccessor<float>(*executor.getAllocator());

    //    EXPECT_FLOAT_EQ(*accessor[0][0], 18.0);
    //    EXPECT_FLOAT_EQ(*accessor[0][1], 18.0);
    //    EXPECT_FLOAT_EQ(*accessor[0][2], 18.0);
//    EXPECT_FLOAT_EQ(*accessor[1][0], 18.0);
//    EXPECT_FLOAT_EQ(*accessor[1][1], 18.0);
//    EXPECT_FLOAT_EQ(*accessor[1][2], 18.0);
//    EXPECT_FLOAT_EQ(*accessor[2][0], 18.0);
//    EXPECT_FLOAT_EQ(*accessor[2][1], 18.0);
//    EXPECT_FLOAT_EQ(*accessor[2][2], 18.0);

    std::cout << "Result: " << *accessor[0][0] << std::endl;
    std::cout << std::endl;
    auto accessorDbg = outputNodeDbg.getAccessor<float>(*executor.getAllocator());
    std::cout << "Weights" << std::endl;
    for (size_t index = 0; index < 9; ++index) {
        std::cout << *accessorDbg[0][index] << ' ';
    }
    std::cout << std::endl;
    /*std::cout << "\n\n@@@\n" << std::endl;
    for (size_t index = 0; index < 9; ++index) {
        std::cout << weights[index] << std::endl;
    }*/
}
