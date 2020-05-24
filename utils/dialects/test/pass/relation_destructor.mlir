// RUN: athena-opt --destroy-graph-relations %s | FileCheck %s
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
    %0 = "ath_graph.create_tensor"() {virtual_address = 9 : index} : () -> tensor<8xf32>
    "ath_graph.alloc"(%0) : (tensor<8xf32>) -> ()
    "ath_graph.lock"(%0) {lock_type = "read_write"} : (tensor<8xf32>) -> ()
    "ath_graph.invoke_loader"(%0) : (tensor<8xf32>) -> ()
    "ath_graph.release"(%0) : (tensor<8xf32>) -> ()
    "ath_graph.return"(%0) : (tensor<8xf32>) -> ()
  }) {cluster_id = 0 : index, node_id = 1 : index, sym_name = "inputB", type = () -> tensor<8xf32>} : () -> ()
  "ath_graph.node"() ( {
  ^bb0(%arg0: tensor<8xf32>, %arg1: tensor<8xf32>):  // no predecessors
    %0 = "ath_graph.create_tensor"() {virtual_address = 17 : index} : () -> tensor<8xf32>
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

// CHECK: module {
// CHECK-NEXT: "ath_graph.node"() ( {
// CHECK-NEXT: %0 = "ath_graph.create_tensor"() {virtual_address = 1 : index} : () -> tensor<8xf32>
// CHECK-NEXT: "ath_graph.alloc"(%0) : (tensor<8xf32>) -> ()
// CHECK-NEXT: "ath_graph.lock"(%0) {lock_type = "read_write"} : (tensor<8xf32>) -> ()
// CHECK-NEXT: "ath_graph.invoke_loader"(%0) : (tensor<8xf32>) -> ()
// CHECK-NEXT: "ath_graph.release"(%0) : (tensor<8xf32>) -> ()
// CHECK-NEXT: ath_graph.return %0 : tensor<8xf32>
// CHECK-NEXT: }) {cluster_id = 0 : index, node_id = 0 : index, sym_name = "inputA", type = () -> tensor<8xf32>} : () -> ()
// CHECK-NEXT: "ath_graph.node"() ( {
// CHECK-NEXT: %0 = "ath_graph.create_tensor"() {virtual_address = 9 : index} : () -> tensor<8xf32>
// CHECK-NEXT: "ath_graph.alloc"(%0) : (tensor<8xf32>) -> ()
// CHECK-NEXT: "ath_graph.lock"(%0) {lock_type = "read_write"} : (tensor<8xf32>) -> ()
// CHECK-NEXT: "ath_graph.invoke_loader"(%0) : (tensor<8xf32>) -> ()
// CHECK-NEXT: "ath_graph.release"(%0) : (tensor<8xf32>) -> ()
// CHECK-NEXT: ath_graph.return %0 : tensor<8xf32>
// CHECK-NEXT: }) {cluster_id = 0 : index, node_id = 1 : index, sym_name = "inputB", type = () -> tensor<8xf32>} : () -> ()
// CHECK-NEXT: "ath_graph.node"() ( {
// CHECK-NEXT: %0 = "ath_graph.create_tensor"() {virtual_address = 9 : index} : () -> tensor<8xf32>
// CHECK-NEXT: %1 = "ath_graph.create_tensor"() {virtual_address = 1 : index} : () -> tensor<8xf32>
// CHECK-NEXT: %2 = "ath_graph.create_tensor"() {virtual_address = 17 : index} : () -> tensor<8xf32>
// CHECK-NEXT: "ath_graph.lock"(%1) {lock_type = "read"} : (tensor<8xf32>) -> ()
// CHECK-NEXT: "ath_graph.lock"(%0) {lock_type = "read"} : (tensor<8xf32>) -> ()
// CHECK-NEXT: "ath_graph.alloc"(%2) : (tensor<8xf32>) -> ()
// CHECK-NEXT: "ath_graph.lock"(%2) {lock_type = "read_write"} : (tensor<8xf32>) -> ()
// CHECK-NEXT: %cst = constant 1.000000e+00 : f32
// CHECK-NEXT: %3 = "ath_graph.add"(%1, %cst, %0, %cst, %2) : (tensor<8xf32>, f32, tensor<8xf32>, f32, tensor<8xf32>) -> tensor<8xf32>
// CHECK-NEXT: "ath_graph.release"(%1) : (tensor<8xf32>) -> ()
// CHECK-NEXT: "ath_graph.release"(%0) : (tensor<8xf32>) -> ()
// CHECK-NEXT: "ath_graph.release"(%2) : (tensor<8xf32>) -> ()
// CHECK-NEXT: ath_graph.return %3 : tensor<8xf32>
// CHECK-NEXT: }) {cluster_id = 1 : index, node_id = 2 : index, sym_name = "sum", type = () -> tensor<8xf32>} : () -> ()
// CHECK-NEXT: "ath_graph.graph"() ( {
// CHECK-NEXT: %0 = ath_graph.eval @inputA() : () -> tensor<8xf32>
// CHECK-NEXT: %1 = ath_graph.eval @inputB() : () -> tensor<8xf32>
// CHECK-NEXT: "ath_graph.barrier"() {clusterId = 0 : index} : () -> ()
// CHECK-NEXT: %2 = ath_graph.eval @sum() : () -> tensor<8xf32>
// CHECK-NEXT: "ath_graph.graph_terminator"() : () -> ()
// CHECK-NEXT: }) {sym_name = "mainGraph", type = () -> ()} : () -> ()
// CHECK-NEXT: }
