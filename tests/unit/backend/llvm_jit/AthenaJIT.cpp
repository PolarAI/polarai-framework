#include <gtest/gtest.h>

#include "../../../../src/backend/llvm/jit/AthenaJIT.h"

#include "AthenaGraph/AthenaGraphDialect.h"
#include "AthenaRuntime/AthenaRuntimeDialect.h"
#include "mlir/InitAllDialects.h"
#include "mlir/InitAllPasses.h"
#include "mlir/IR/Dialect.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/Pass/Pass.h"
#include "llvm/Support/InitLLVM.h"
#include "llvm/Support/TargetSelect.h"

using namespace athena::backend::llvm;

static constexpr auto IR = R"(
module {
  "ath_graph.node"() ( {
    %0 = "ath_graph.create_tensor"() {virtual_address = 1 : index} : () -> tensor<8xf32>
    "ath_graph.alloc"(%0) : (tensor<8xf32>) -> ()
    "ath_graph.lock"(%0) {lock_type = "read_write"} : (tensor<8xf32>) -> ()
    "ath_graph.invoke_loader"(%0) : (tensor<8xf32>) -> ()
    "ath_graph.release"(%0) : (tensor<8xf32>) -> ()
    ath_graph.return %0 : tensor<8xf32>
  }) {cluster_id = 0 : index, node_id = 0 : index, sym_name = "inputA", type = () -> tensor<8xf32>} : () -> ()
  "ath_graph.node"() ( {
    %0 = "ath_graph.create_tensor"() {virtual_address = 9 : index} : () -> tensor<8xf32>
    "ath_graph.alloc"(%0) : (tensor<8xf32>) -> ()
    "ath_graph.lock"(%0) {lock_type = "read_write"} : (tensor<8xf32>) -> ()
    "ath_graph.invoke_loader"(%0) : (tensor<8xf32>) -> ()
    "ath_graph.release"(%0) : (tensor<8xf32>) -> ()
    ath_graph.return %0 : tensor<8xf32>
  }) {cluster_id = 0 : index, node_id = 1 : index, sym_name = "inputB", type = () -> tensor<8xf32>} : () -> ()
  "ath_graph.node"() ( {
    %0 = "ath_graph.create_tensor"() {virtual_address = 9 : index} : () -> tensor<8xf32>
    %1 = "ath_graph.create_tensor"() {virtual_address = 1 : index} : () -> tensor<8xf32>
    %2 = "ath_graph.create_tensor"() {virtual_address = 17 : index} : () -> tensor<8xf32>
    "ath_graph.lock"(%1) {lock_type = "read"} : (tensor<8xf32>) -> ()
    "ath_graph.lock"(%0) {lock_type = "read"} : (tensor<8xf32>) -> ()
    "ath_graph.alloc"(%2) : (tensor<8xf32>) -> ()
    "ath_graph.lock"(%2) {lock_type = "read_write"} : (tensor<8xf32>) -> ()
    %cst = constant 1.000000e+00 : f32
    %3 = "ath_graph.add"(%1, %cst, %0, %cst, %2) : (tensor<8xf32>, f32, tensor<8xf32>, f32, tensor<8xf32>) -> tensor<8xf32>
    "ath_graph.release"(%1) : (tensor<8xf32>) -> ()
    "ath_graph.release"(%0) : (tensor<8xf32>) -> ()
    "ath_graph.release"(%2) : (tensor<8xf32>) -> ()
    ath_graph.return %3 : tensor<8xf32>
  }) {cluster_id = 1 : index, node_id = 2 : index, sym_name = "sum", type = () -> tensor<8xf32>} : () -> ()
  "ath_graph.graph"() ( {
    %0 = ath_graph.eval @inputA() : () -> tensor<8xf32>
    %1 = ath_graph.eval @inputB() : () -> tensor<8xf32>
    "ath_graph.barrier"() {clusterId = 0 : index} : () -> ()
    %2 = ath_graph.eval @sum() : () -> tensor<8xf32>
    "ath_graph.graph_terminator"() : () -> ()
  }) {sym_name = "mainGraph", type = () -> ()} : () -> ()
}
)";

TEST(JITTest, CompilesIRCorrectly) {
  mlir::registerAllDialects();
  mlir::registerAllPasses();

  mlir::registerDialect<mlir::ath_graph::AthenaGraphDialect>();
  mlir::registerDialect<mlir::ath_rt::AthenaRuntimeDialect>();

  ::llvm::InitializeNativeTarget();
  ::llvm::InitializeNativeTargetAsmPrinter();

  mlir::MLIRContext context;
  auto module = mlir::parseSourceString(IR, &context);

  auto JIT = AthenaJIT::create();
  JIT->addModule(module);

  auto mainGraphFunc = JIT->lookupSymbol("mainGraph");
}
