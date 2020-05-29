#include <gtest/gtest.h>

#include "../../../../src/backend/llvm/jit/AthenaJIT.h"

#include "AthenaGraph/AthenaGraphDialect.h"
#include "AthenaRuntime/AthenaRuntimeDialect.h"
#include "mlir/IR/Dialect.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/InitAllDialects.h"
#include "mlir/InitAllPasses.h"
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

int allocateCalls = 0;
int releaseCalls = 0;
int lockCalls = 0;
int loadCalls = 0;
int selectCalls = 0;
int barrierCalls = 0;
int launchCalls = 0;

extern "C" {
void ath_allocate(void*, void*, void*) { allocateCalls++; }
void ath_release(void*, void*, void*) { releaseCalls++; }
void ath_lock(void*, void*, void*, int) { lockCalls++; }
void* ath_device_select(void*, uint64_t) { selectCalls++; return nullptr; }
void ath_load(void*, uint64_t, void*) { loadCalls++; }
void ath_barrier(uint64_t, void*) { barrierCalls++; }
void* ath_launch(void*, void*, void*, void*) { launchCalls++; return nullptr; }
}

TEST(JITTest, CompilesIRCorrectly) {
  mlir::registerAllDialects();
  mlir::registerAllPasses();

  mlir::registerDialect<mlir::ath_graph::AthenaGraphDialect>();
  mlir::registerDialect<mlir::ath_rt::AthenaRuntimeDialect>();

  ::llvm::InitializeNativeTarget();
  ::llvm::InitializeNativeTargetAsmPrinter();

  auto JIT = AthenaJIT::create();
  auto module = mlir::parseSourceString(IR, JIT->getContext());

  JIT->addModule(module);

  auto mainGraphSym = JIT->lookupSymbol("mainGraph");
  auto mainGraphFunc = reinterpret_cast<void (*)(void*)>(mainGraphSym);
  mainGraphFunc(nullptr);

  EXPECT_EQ(allocateCalls, 3);
  EXPECT_EQ(lockCalls, 5);
  EXPECT_EQ(releaseCalls, 5);
  EXPECT_EQ(selectCalls, 14);
  EXPECT_EQ(barrierCalls, 1);
  EXPECT_EQ(launchCalls, 1);
}
