//===----------------------------------------------------------------------===//
// Copyright (c) 2020 Athena. All rights reserved.
// https://getathena.ml
//
// Licensed under MIT license.
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS, WITHOUT
// WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the
// License for the specific language governing permissions and limitations under
// the License.
//===----------------------------------------------------------------------===//

#include <athena/core/DataType.h>
#include <athena/core/inner/Tensor.h>
#include <athena/backend/llvm/CodeGen.h>
#include <athena/core/Generator.h>

#include "llvm/Support/raw_ostream.h"
#include <mlir/IR/Builders.h>
#include <mlir/IR/MLIRContext.h>
#include <mlir/IR/Module.h>

#include <effcee/effcee.h>
#include <gtest/gtest.h>

#include <any>
#include <fstream>
#include <utility>
#include <vector>

using ::testing::Test;
using namespace athena::core;
using namespace athena::backend::llvm;

constexpr static int tensorSize = 8;

static GenNode createInputNode(Context& ctx, std::string_view name,
                               size_t nodeId,
                               inner::Tensor& outValue,
                                                         Generator& generator) {
  std::vector<inner::Tensor> args;
  GenNode node = generator.createNode(name, nodeId, 0, args, outValue);

  auto save = generator.getInsertionPoint();
  generator.setInsertionPoint(node);
  generator.callBuiltin<builtin::Alloc>(node.getResult());
  // generator.callBuiltin(builtin::Lock, node.getResult(), LockType::READ_WRITE);
  // generator.callBuiltin(builtin::InvokeLoader, node.getResult(), name);
  // generator.callBuiltin(builtin::Release, node.getResult());
  generator.setInsertionPoint(save);

  return node;
}

TEST(MLIRRegression, BasicIR) {
  mlir::MLIRContext context;
  mlir::OpBuilder builder(&context);
  auto module = builder.create<mlir::ModuleOp>(builder.getUnknownLoc());
  builder.setInsertionPointToStart(module.getBody());

  Generator generator;
  populateCodeGenPatterns(generator, builder);

  Context ctx;

  inner::Tensor tensorA(DataType::FLOAT, {tensorSize}, ctx);
  inner::Tensor tensorB(DataType::FLOAT, {tensorSize}, ctx);
  auto nodeA = createInputNode(ctx, "inputA", 0, tensorA, generator);
  // auto nodeB = createInputNode(ctx, "inputB", 1,tensorB, generator);

  // std::vector<inner::Tensor> args{tensorA, tensorB};
  // inner::Tensor tensorC(DataType::FLOAT, {tensorSize}, ctx);
  // auto nodeC = generator.createNode("sum", 2, 1, args, tensorC);

  // auto save = generator.getInsertionPoint();
  // generator.setInsertionPoint(nodeC);
  // generator.callBuiltin(builtin::Lock, nodeC.getOperand(0), LockType::READ);
  // generator.callBuiltin(builtin::Lock, nodeC.getOperand(1), LockType::READ);

  // generator.callBuiltin(builtin::Alloc, nodeC.getResult());
  // generator.callBuiltin(builtin::Lock, nodeC.getResult(), LockType::READ_WRITE);

  // auto one = generator.createConstant(1.0f);
  // generator.callBuiltin(builtin::Add, nodeC.getOperand(0), one,
  //                       nodeC.getOperand(1), one, nodeC.getResult());

  // generator.callBuiltin(builtin::Release, nodeC.getOperand(0));
  // generator.callBuiltin(builtin::Release, nodeC.getOperand(1));
  // generator.callBuiltin(builtin::Release, nodeC.getResult());

  // generator.setInsertionPoint(save);

  // auto graph = generator.createGraph("mainGraph", 0);
  // generator.setInsertionPoint(graph);

  // std::vector<GenValue> empty;
  // auto resA = generator.callBuiltin(builtin::NodeEval, graph, nodeA, empty);
  // auto resB = generator.callBuiltin(builtin::NodeEval, graph, nodeB, empty);

  // std::vector<GenValue> cArgs{resA, resB};
  // generator.callBuiltin(builtin::NodeEval, graph, nodeC, cArgs);

  /*auto result =
      effcee::Match(str, matches,
  effcee::Options().SetChecksName("checks"));

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
  }*/
  SUCCEED();
}