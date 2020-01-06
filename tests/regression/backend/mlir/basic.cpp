/*
 * Copyright (c) 2020 Athena. All rights reserved.
 * https://getathena.ml
 *
 * Licensed under MIT license.
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS, WITHOUT
 * WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the
 * License for the specific language governing permissions and limitations under
 * the License.
 */

#include <athena/backend/llvm/LLVMExecutor.h>
#include <athena/backend/llvm/LLVMTrivialAllocator.h>
#include <athena/backend/llvm/mlir/MLIRGenerator.h>
#include <athena/core/GradientDescent.h>
#include <athena/core/Graph.h>
#include <athena/core/GraphCompiler.h>
#include <athena/core/InputNode.h>
#include <athena/core/LossNode.h>
#include <athena/core/Node.h>
#include <athena/core/inner/Tensor.h>
#include <athena/loaders/MemoryLoader/MemoryLoader.h>
#include <athena/ops/AddOperation.h>
#include <athena/ops/MSELossFunction.h>

#include "llvm/Support/raw_ostream.h"

#include <effcee/effcee.h>
#include <fstream>
#include <gtest/gtest.h>

using namespace athena::core;
using namespace athena::ops;
using namespace athena::backend::llvm;
using namespace athena::loaders;
using ::testing::Test;

std::string matches =
    "CHECK: func @evaluate()\n"
    "CHECK-NEXT: %0 = \"graph.alloca\"() {cluster_id = 0 : i64, node_id = 1 : "
    "i64, node_name = \"temp_node_name\", tensor_addr = 4 : i64} : () -> "
    "tensor<3xf32>\n"
    "CHECK-NEXT: %1 = \"graph.alloca\"() {cluster_id = 0 : i64, node_id = 1 : "
    "i64, node_name = \"temp_node_name\", tensor_addr = 1 : i64} : () -> "
    "tensor<3xf32>\n"
    "CHECK-NEXT: %2 = \"graph.alloca\"() {cluster_id = 0 : i64, node_id = 1 : "
    "i64, node_name = \"temp_node_name\", tensor_addr = 7 : i64} : () -> "
    "tensor<3xf32>\n"
    "CHECK-NEXT: %3 = \"graph.add\"(%1, %0) {cluster_id = 0 : i64, node_id = 1 "
    ": i64, node_name = \"temp_node_name\", tensor_addr = 7 : i64} : "
    "(tensor<3xf32>, tensor<3xf32>) -> tensor<3xf32>\n"
    "CHECK-NEXT: \"graph.return\"() : () -> ()\n";

TEST(MLIRRegression, BasicIR) {
#ifndef DEBUG
  SUCCEED() << "Dumping IR is not supported in Release mode\n";
#else

  // Arrange
  TensorShape shape({3});

  float aData[] = {1, 2, 3};
  float bData[] = {4, 5, 6};
  float cData[] = {0, 0, 0};

  MemoryLoader aLoader(aData, 3 * sizeof(float));
  MemoryLoader bLoader(bData, 3 * sizeof(float));
  MemoryLoader cLoader(cData, 3 * sizeof(float));

  Context context;
  Graph graph(context);
  graph.setUpOptimizer<Optimizer>(/*learningRate0.01*/);
  graph.setUpOptimizer<GradientDescent>(/*learningRate*/ 0.01);
  InputNode aInp(shape, DataType::FLOAT, aLoader, context, false, "a");
  InputNode bInp(shape, DataType::FLOAT, bLoader, context, false, "b");
  graph.addNode(aInp);
  graph.addNode(bInp);

  AddOperation addOp;
  Node add(addOp, context, "vector_add_1");
  graph.addNode(add);
  add.after(aInp, 1);
  add.after(bInp, 2);

  OutputNode outputNode(DataType::FLOAT, context, "out");
  graph.addNode(outputNode);
  outputNode.after(add, 1);

  // Act
  MLIRGenerator generator;

  GraphCompiler::compileForward(graph, generator);
  std::string str;
  llvm::raw_string_ostream stringOstream(str);
  generator.getModule().print(stringOstream);
  stringOstream.str();

  // Assert
  auto result =
      effcee::Match(str, matches, effcee::Options().SetChecksName("checks"));

  if (result) {
    SUCCEED();
  } else {
    // Otherwise, you can get a status code and a detailed message.
    switch (result.status()) {
    case effcee::Result::Status::NoRules:
      std::cout << "error: Expected check rules\n";
      break;
    case effcee::Result::Status::Fail:
      std::cout << "The input failed to match check rules:\n";
      break;
    default:
      break;
    }
    std::cout << result.message() << std::endl;
    FAIL();
  }
#endif
}