// RUN: athena-opt --deploy-default-functions %s | FileCheck %s
module {
  "ath_graph.node"() ( {
  ^bb0(%arg0: index, %arg1: index):	// no predecessors
    %0 = "ath_graph.get_tensor"(%arg0) {virtual_address = 1 : index} : (index) -> tensor<8xf32>
    "ath_graph.alloc"(%0) : (tensor<8xf32>) -> ()
    "ath_graph.lock"(%0) {lock_type = "read_write"} : (tensor<8xf32>) -> ()
    // CHECK-DAG: llvm.func @MyLoaderLoad(!llvm<"i8*">, !llvm<"i8*">, !llvm<"i8*">)
    "ath_graph.invoke_loader"(%0) {loader_routine = "MyLoaderLoad"} : (tensor<8xf32>) -> ()
    "ath_graph.release"(%0) : (tensor<8xf32>) -> ()
    ath_graph.return %0 : tensor<8xf32>
  }) {sym_name = "InputA", type = (index, index) -> tensor<8xf32>} : () -> ()
  "ath_graph.node"() ( {
  ^bb0(%arg0: index, %arg1: index):	// no predecessors
    %0 = "ath_graph.get_tensor"(%arg0) {virtual_address = 9 : index} : (index) -> tensor<8xf32>
    "ath_graph.alloc"(%0) : (tensor<8xf32>) -> ()
    "ath_graph.lock"(%0) {lock_type = "read_write"} : (tensor<8xf32>) -> ()
    // CHECK-DAG: llvm.func @SuperLoaderLoad(!llvm<"i8*">, !llvm<"i8*">, !llvm<"i8*">)
    "ath_graph.invoke_loader"(%0) {loader_routine = "SuperLoaderLoad"} : (tensor<8xf32>) -> ()
    "ath_graph.release"(%0) : (tensor<8xf32>) -> ()
    ath_graph.return %0 : tensor<8xf32>
  }) {sym_name = "InputB", type = (index, index) -> tensor<8xf32>} : () -> ()
}
