#include <gtest/gtest.h>

#include "../../../../src/backend/llvm/jit/AthenaJIT.h"

#include "AthenaGraph/AthenaGraphDialect.h"
#include "AthenaRuntime/AthenaRuntimeDialect.h"
#include "AthenaGraph/AthenaGraphDialect.h"
#include "AthenaRuntime/AthenaRuntimeDialect.h"
#include "Conversion/GraphToRuntimePass.h"
#include "Conversion/RuntimeToLLVM.h"
#include "Passes/Passes.h"

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
"ath_graph.return"(%0) : (tensor<8xf32>) -> ()
}) {cluster_id = 0 : index, node_id = 0 : index, sym_name = "inputA", type = () -> tensor<8xf32>} : () -> ()
"ath_graph.node"() ( {
%0 = "ath_graph.create_tensor"() {virtual_address = 33 : index} : () -> tensor<8xf32>
"ath_graph.alloc"(%0) : (tensor<8xf32>) -> ()
"ath_graph.lock"(%0) {lock_type = "read_write"} : (tensor<8xf32>) -> ()
"ath_graph.invoke_loader"(%0) : (tensor<8xf32>) -> ()
"ath_graph.release"(%0) : (tensor<8xf32>) -> ()
"ath_graph.return"(%0) : (tensor<8xf32>) -> ()
}) {cluster_id = 0 : index, node_id = 1 : index, sym_name = "inputB", type = () -> tensor<8xf32>} : () -> ()
"ath_graph.node"() ( {
^bb0(%arg0: tensor<8xf32>, %arg1: tensor<8xf32>):  // no predecessors
%0 = "ath_graph.create_tensor"() {virtual_address = 65 : index} : () -> tensor<8xf32>
"ath_graph.lock"(%arg0) {lock_type = "read"} : (tensor<8xf32>) -> ()
"ath_graph.lock"(%arg1) {lock_type = "read"} : (tensor<8xf32>) -> ()
"ath_graph.alloc"(%0) : (tensor<8xf32>) -> ()
"ath_graph.lock"(%0) {lock_type = "read_write"} : (tensor<8xf32>) -> ()
%1 = "std.constant"() {value = 1.000000e+00 : f32} : () -> f32
%2 = "ath_graph.add"(%arg0, %1, %arg1, %1, %0) : (tensor<8xf32>, f32, tensor<8xf32>, f32, tensor<8xf32>) -> tensor<8xf32>
"ath_graph.release"(%arg0) : (tensor<8xf32>) -> ()
"ath_graph.release"(%arg1) : (tensor<8xf32>) -> ()
"ath_graph.release"(%0) : (tensor<8xf32>) -> ()
"ath_graph.return"(%2) : (tensor<8xf32>) -> ()
}) {cluster_id = 1 : index, node_id = 2 : index, sym_name = "sum", type = (tensor<8xf32>, tensor<8xf32>) -> tensor<8xf32>} : () -> ()
"ath_graph.graph"() ( {
%0 = "ath_graph.eval"() {node = @inputA} : () -> tensor<8xf32>
%1 = "ath_graph.eval"() {node = @inputB} : () -> tensor<8xf32>
"ath_graph.barrier"() {clusterId = 0 : index} : () -> ()
%2 = "ath_graph.eval"(%0, %1) {node = @sum} : (tensor<8xf32>, tensor<8xf32>) -> tensor<8xf32>
"ath_graph.graph_terminator"() : () -> ()
}) {sym_name = "mainGraph", type = () -> ()} : () -> ()
}
)";

TEST(JITTest, CompilesIRCorrectly) {
  mlir::registerAllDialects();
  mlir::registerAllPasses();


  mlir::registerDialect<mlir::ath_graph::AthenaGraphDialect>();
  mlir::registerDialect<mlir::ath_rt::AthenaRuntimeDialect>();
  mlir::registerPass("convert-graph-to-runtime",
                     "Converts Athena Graph to Runtime calls",
                     mlir::createLowerGraphToRuntimePass);
  mlir::registerPass("convert-runtime-to-llvm",
                     "Converts Athena Graph to Runtime calls",
                     mlir::createLowerRuntimeToLLVMPass);
  mlir::registerPass("deploy-default-functions",
                     "Adds definitions of default Athena functions",
                     mlir::createDeployDefaultFunctionsPass);
  mlir::registerPass("destroy-graph-relations",
                     "Removes explicit dependencies between Graph nodes",
                     mlir::createGraphRelationDestructorPass);
  mlir::registerPass("legalize-barriers",
                     "Adds event arguments to Runtime barriers",
                     mlir::createBarrierLegalizerPass);

  ::llvm::InitializeNativeTarget();
  ::llvm::InitializeNativeTargetAsmPrinter();

  mlir::MLIRContext context;
  auto module = mlir::parseSourceString(IR, &context);

  auto JIT = AthenaJIT::create();
  JIT->addModule(module);

  auto mainGraphFunc = JIT->lookupSymbol("mainGraph");
}
