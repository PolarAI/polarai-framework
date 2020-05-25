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
//
#include <athena/backend/llvm/CodeGen.h>
#include <athena/core/Generator.h>
#include <athena/core/tensor/DataType.h>
#include <athena/core/tensor/internal/TensorInternal.h>
#include <athena/core/context/Context.h>

#include <llvm/Support/raw_ostream.h>
#include <llvm/Support/Debug.h>
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
using namespace athena::core::internal;
using namespace athena::backend::llvm;

constexpr static int tensorSize = 8;

static constexpr auto checks = R"(
// CHECK: module {
// CHECK-NEXT: "ath_graph.node"() ( {
// CHECK-NEXT: %0 = "ath_graph.create_tensor"() {virtual_address = 1 : index} : () -> tensor<8xf32>
// CHECK-NEXT: "ath_graph.alloc"(%0) : (tensor<8xf32>) -> ()
// CHECK-NEXT: "ath_graph.lock"(%0) {lock_type = "read_write"} : (tensor<8xf32>) -> ()
// CHECK-NEXT: "ath_graph.invoke_loader"(%0) : (tensor<8xf32>) -> ()
// CHECK-NEXT: "ath_graph.release"(%0) : (tensor<8xf32>) -> ()
// CHECK-NEXT: "ath_graph.return"(%0) : (tensor<8xf32>) -> ()
// CHECK-NEXT: }) {cluster_id = 0 : index, node_id = 0 : index, sym_name = "inputA", type = () -> tensor<8xf32>} : () -> ()
// CHECK-NEXT: "ath_graph.node"() ( {
// CHECK-NEXT: %0 = "ath_graph.create_tensor"() {virtual_address = 33 : index} : () -> tensor<8xf32>
// CHECK-NEXT: "ath_graph.alloc"(%0) : (tensor<8xf32>) -> ()
// CHECK-NEXT: "ath_graph.lock"(%0) {lock_type = "read_write"} : (tensor<8xf32>) -> ()
// CHECK-NEXT: "ath_graph.invoke_loader"(%0) : (tensor<8xf32>) -> ()
// CHECK-NEXT: "ath_graph.release"(%0) : (tensor<8xf32>) -> ()
// CHECK-NEXT: "ath_graph.return"(%0) : (tensor<8xf32>) -> ()
// CHECK-NEXT: }) {cluster_id = 0 : index, node_id = 1 : index, sym_name = "inputB", type = () -> tensor<8xf32>} : () -> ()
// CHECK-NEXT: "ath_graph.node"() ( {
// CHECK-NEXT: ^bb0(%arg0: tensor<8xf32>, %arg1: tensor<8xf32>):  // no predecessors
// CHECK-NEXT: %0 = "ath_graph.create_tensor"() {virtual_address = 65 : index} : () -> tensor<8xf32>
// CHECK-NEXT: "ath_graph.lock"(%arg0) {lock_type = "read"} : (tensor<8xf32>) -> ()
// CHECK-NEXT: "ath_graph.lock"(%arg1) {lock_type = "read"} : (tensor<8xf32>) -> ()
// CHECK-NEXT: "ath_graph.alloc"(%0) : (tensor<8xf32>) -> ()
// CHECK-NEXT: "ath_graph.lock"(%0) {lock_type = "read_write"} : (tensor<8xf32>) -> ()
// CHECK-NEXT: %1 = "std.constant"() {value = 1.000000e+00 : f32} : () -> f32
// CHECK-NEXT: %2 = "ath_graph.add"(%arg0, %1, %arg1, %1, %0) : (tensor<8xf32>, f32, tensor<8xf32>, f32, tensor<8xf32>) -> tensor<8xf32>
// CHECK-NEXT: "ath_graph.release"(%arg0) : (tensor<8xf32>) -> ()
// CHECK-NEXT: "ath_graph.release"(%arg1) : (tensor<8xf32>) -> ()
// CHECK-NEXT: "ath_graph.release"(%0) : (tensor<8xf32>) -> ()
// CHECK-NEXT: "ath_graph.return"(%2) : (tensor<8xf32>) -> ()
// CHECK-NEXT: }) {cluster_id = 1 : index, node_id = 2 : index, sym_name = "sum", type = (tensor<8xf32>, tensor<8xf32>) -> tensor<8xf32>} : () -> ()
// CHECK-NEXT: "ath_graph.graph"() ( {
// CHECK-NEXT: %0 = "ath_graph.eval"() {node = @inputA} : () -> tensor<8xf32>
// CHECK-NEXT: %1 = "ath_graph.eval"() {node = @inputB} : () -> tensor<8xf32>
// CHECK-NEXT: "ath_graph.barrier"() {clusterId = 0 : index} : () -> ()
// CHECK-NEXT: %2 = "ath_graph.eval"(%0, %1) {node = @sum} : (tensor<8xf32>, tensor<8xf32>) -> tensor<8xf32>
// CHECK-NEXT: "ath_graph.graph_terminator"() : () -> ()
// CHECK-NEXT: }) {sym_name = "mainGraph", type = () -> ()} : () -> ()
// CHECK-NEXT: }
)";

static GenNode createInputNode(Context& ctx, std::string_view name,
                               size_t nodeId, internal::TensorInternal& outValue,
                               Generator& generator) {
  std::vector<internal::TensorInternal*> args;
  GenNode node = generator.createNode(name, nodeId, 0, args, outValue);

  auto save = generator.getInsertionPoint();
  generator.setInsertionPoint(node);
  generator.callBuiltin<builtin::Alloc>(node.getResult());
  generator.callBuiltin<builtin::Lock>(node.getResult(), LockType::READ_WRITE);
  generator.callBuiltin<builtin::InvokeLoader>(node.getResult());
  generator.callBuiltin<builtin::Release>(node.getResult());
  generator.callBuiltin<builtin::Return>(node.getResult());
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
  auto ctxInternal = ctx.internal();

  auto tensorAId = ctxInternal->create<internal::TensorInternal>(ctxInternal, ctxInternal->getNextPublicIndex(), DataType::FLOAT, TensorShape{tensorSize});
  auto tensorA = ctxInternal->getRef<internal::TensorInternal>(tensorAId);
  auto tensorBId = ctxInternal->create<internal::TensorInternal>(ctxInternal, ctxInternal->getNextPublicIndex(), DataType::FLOAT, TensorShape{tensorSize});
  auto tensorB = ctxInternal->getRef<internal::TensorInternal>(tensorBId);
  auto nodeA = createInputNode(ctx, "inputA", 0, tensorA, generator);
  auto nodeB = createInputNode(ctx, "inputB", 1, tensorB, generator);

  std::vector<internal::TensorInternal*> args{&tensorA, &tensorB};
  auto tensorCId = ctxInternal->create<internal::TensorInternal>(ctxInternal, ctxInternal->getNextPublicIndex(), DataType::FLOAT, TensorShape{tensorSize});
  auto tensorC = ctxInternal->getRef<internal::TensorInternal>(tensorCId);
  auto nodeC = generator.createNode("sum", 2, 1, args, tensorC);

  auto save = generator.getInsertionPoint();
  generator.setInsertionPoint(nodeC);
  generator.callBuiltin<builtin::Lock>(nodeC.getOperand(0), LockType::READ);
  generator.callBuiltin<builtin::Lock>(nodeC.getOperand(1), LockType::READ);

  generator.callBuiltin<builtin::Alloc>(nodeC.getResult());
  generator.callBuiltin<builtin::Lock>(nodeC.getResult(), LockType::READ_WRITE);

  auto one = generator.createConstant(1.0f);
  auto res = generator.callBuiltin<builtin::Add>(
      nodeC.getOperand(0), one, nodeC.getOperand(1), one, nodeC.getResult());

  generator.callBuiltin<builtin::Release>(nodeC.getOperand(0));
  generator.callBuiltin<builtin::Release>(nodeC.getOperand(1));
  generator.callBuiltin<builtin::Release>(nodeC.getResult());

  generator.callBuiltin<builtin::Return>(res);

  generator.setInsertionPoint(save);

  auto graph = generator.createGraph("mainGraph", 0);
  generator.setInsertionPoint(graph);

  std::vector<GenValue> empty;
  auto resA = generator.callBuiltin<builtin::NodeEval>(graph, nodeA, empty);
  auto resB = generator.callBuiltin<builtin::NodeEval>(graph, nodeB, empty);

  generator.callBuiltin<builtin::Barrier>(0);

  std::vector<GenValue> cArgs{resA, resB};
  generator.callBuiltin<builtin::NodeEval>(graph, nodeC, cArgs);

  std::string str;
  ::llvm::raw_string_ostream stream(str);
  module.print(stream);
  auto result =
      effcee::Match(str, checks,
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
  }
}
