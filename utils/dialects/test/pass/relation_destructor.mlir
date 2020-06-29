// RUN: polar-opt --destroy-graph-relations %s | FileCheck %s
module {
  "polar_graph.node"() ( {
    %0 = "polar_graph.create_tensor"() {virtual_address = 1 : index} : () -> tensor<8xf32>
    "polar_graph.alloc"(%0) : (tensor<8xf32>) -> ()
    "polar_graph.lock"(%0) {lock_type = "read_write"} : (tensor<8xf32>) -> ()
    "polar_graph.invoke_loader"(%0) : (tensor<8xf32>) -> ()
    "polar_graph.release"(%0) : (tensor<8xf32>) -> ()
    "polar_graph.return"(%0) : (tensor<8xf32>) -> ()
  }) {cluster_id = 0 : index, node_id = 0 : index, sym_name = "inputA", type = () -> tensor<8xf32>} : () -> ()
  "polar_graph.node"() ( {
    %0 = "polar_graph.create_tensor"() {virtual_address = 9 : index} : () -> tensor<8xf32>
    "polar_graph.alloc"(%0) : (tensor<8xf32>) -> ()
    "polar_graph.lock"(%0) {lock_type = "read_write"} : (tensor<8xf32>) -> ()
    "polar_graph.invoke_loader"(%0) : (tensor<8xf32>) -> ()
    "polar_graph.release"(%0) : (tensor<8xf32>) -> ()
    "polar_graph.return"(%0) : (tensor<8xf32>) -> ()
  }) {cluster_id = 0 : index, node_id = 1 : index, sym_name = "inputB", type = () -> tensor<8xf32>} : () -> ()
  "polar_graph.node"() ( {
  ^bb0(%arg0: tensor<8xf32>, %arg1: tensor<8xf32>):  // no predecessors
    %0 = "polar_graph.create_tensor"() {virtual_address = 17 : index} : () -> tensor<8xf32>
    "polar_graph.lock"(%arg0) {lock_type = "read"} : (tensor<8xf32>) -> ()
    "polar_graph.lock"(%arg1) {lock_type = "read"} : (tensor<8xf32>) -> ()
    "polar_graph.alloc"(%0) : (tensor<8xf32>) -> ()
    "polar_graph.lock"(%0) {lock_type = "read_write"} : (tensor<8xf32>) -> ()
    %1 = "std.constant"() {value = 1.000000e+00 : f32} : () -> f32
    "polar_graph.add"(%arg0, %1, %arg1, %1, %0) : (tensor<8xf32>, f32, tensor<8xf32>, f32, tensor<8xf32>) -> ()
    "polar_graph.release"(%arg0) : (tensor<8xf32>) -> ()
    "polar_graph.release"(%arg1) : (tensor<8xf32>) -> ()
    "polar_graph.release"(%0) : (tensor<8xf32>) -> ()
    "polar_graph.return"(%0) : (tensor<8xf32>) -> ()
  }) {cluster_id = 1 : index, node_id = 2 : index, sym_name = "sum", type = (tensor<8xf32>, tensor<8xf32>) -> tensor<8xf32>} : () -> ()
  "polar_graph.graph"() ( {
    %0 = "polar_graph.eval"() {node = @inputA} : () -> tensor<8xf32>
    %1 = "polar_graph.eval"() {node = @inputB} : () -> tensor<8xf32>
    "polar_graph.barrier"() {clusterId = 0 : index} : () -> ()
    %2 = "polar_graph.eval"(%0, %1) {node = @sum} : (tensor<8xf32>, tensor<8xf32>) -> tensor<8xf32>
    "polar_graph.graph_terminator"() : () -> ()
  }) {sym_name = "mainGraph", type = () -> ()} : () -> ()
}

// CHECK: module {
// CHECK-NEXT: "polar_graph.node"() ( {
// CHECK-NEXT: %0 = "polar_graph.create_tensor"() {virtual_address = 1 : index} : () -> tensor<8xf32>
// CHECK-NEXT: "polar_graph.alloc"(%0) : (tensor<8xf32>) -> ()
// CHECK-NEXT: "polar_graph.lock"(%0) {lock_type = "read_write"} : (tensor<8xf32>) -> ()
// CHECK-NEXT: "polar_graph.invoke_loader"(%0) : (tensor<8xf32>) -> ()
// CHECK-NEXT: "polar_graph.release"(%0) : (tensor<8xf32>) -> ()
// CHECK-NEXT: polar_graph.return %0 : tensor<8xf32>
// CHECK-NEXT: }) {cluster_id = 0 : index, node_id = 0 : index, sym_name = "inputA", type = () -> tensor<8xf32>} : () -> ()
// CHECK-NEXT: "polar_graph.node"() ( {
// CHECK-NEXT: %0 = "polar_graph.create_tensor"() {virtual_address = 9 : index} : () -> tensor<8xf32>
// CHECK-NEXT: "polar_graph.alloc"(%0) : (tensor<8xf32>) -> ()
// CHECK-NEXT: "polar_graph.lock"(%0) {lock_type = "read_write"} : (tensor<8xf32>) -> ()
// CHECK-NEXT: "polar_graph.invoke_loader"(%0) : (tensor<8xf32>) -> ()
// CHECK-NEXT: "polar_graph.release"(%0) : (tensor<8xf32>) -> ()
// CHECK-NEXT: polar_graph.return %0 : tensor<8xf32>
// CHECK-NEXT: }) {cluster_id = 0 : index, node_id = 1 : index, sym_name = "inputB", type = () -> tensor<8xf32>} : () -> ()
// CHECK-NEXT: "polar_graph.node"() ( {
// CHECK-NEXT: %0 = "polar_graph.create_tensor"() {virtual_address = 9 : index} : () -> tensor<8xf32>
// CHECK-NEXT: %1 = "polar_graph.create_tensor"() {virtual_address = 1 : index} : () -> tensor<8xf32>
// CHECK-NEXT: %2 = "polar_graph.create_tensor"() {virtual_address = 17 : index} : () -> tensor<8xf32>
// CHECK-NEXT: "polar_graph.lock"(%1) {lock_type = "read"} : (tensor<8xf32>) -> ()
// CHECK-NEXT: "polar_graph.lock"(%0) {lock_type = "read"} : (tensor<8xf32>) -> ()
// CHECK-NEXT: "polar_graph.alloc"(%2) : (tensor<8xf32>) -> ()
// CHECK-NEXT: "polar_graph.lock"(%2) {lock_type = "read_write"} : (tensor<8xf32>) -> ()
// CHECK-NEXT: %cst = constant 1.000000e+00 : f32
// CHECK-NEXT: "polar_graph.add"(%1, %cst, %0, %cst, %2) : (tensor<8xf32>, f32, tensor<8xf32>, f32, tensor<8xf32>) -> ()
// CHECK-NEXT: "polar_graph.release"(%1) : (tensor<8xf32>) -> ()
// CHECK-NEXT: "polar_graph.release"(%0) : (tensor<8xf32>) -> ()
// CHECK-NEXT: "polar_graph.release"(%2) : (tensor<8xf32>) -> ()
// CHECK-NEXT: polar_graph.return %2 : tensor<8xf32>
// CHECK-NEXT: }) {cluster_id = 1 : index, node_id = 2 : index, sym_name = "sum", type = () -> tensor<8xf32>} : () -> ()
// CHECK-NEXT: "polar_graph.graph"() ( {
// CHECK-NEXT: %0 = polar_graph.eval @inputA() : () -> tensor<8xf32>
// CHECK-NEXT: %1 = polar_graph.eval @inputB() : () -> tensor<8xf32>
// CHECK-NEXT: "polar_graph.barrier"() {clusterId = 0 : index} : () -> ()
// CHECK-NEXT: %2 = polar_graph.eval @sum() : () -> tensor<8xf32>
// CHECK-NEXT: "polar_graph.graph_terminator"() : () -> ()
// CHECK-NEXT: }) {sym_name = "mainGraph", type = () -> ()} : () -> ()
// CHECK-NEXT: }
