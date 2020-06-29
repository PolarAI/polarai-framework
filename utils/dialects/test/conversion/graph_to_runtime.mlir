// RUN: polar-opt --deploy-default-functions --convert-graph-to-runtime --canonicalize %s | FileCheck %s
// XFAIL: *

module {
  "polar_graph.node"() ( {
    %0 = "polar_graph.create_tensor"() {virtual_address = 1 : index} : () -> tensor<2x2xf32>
    "polar_graph.alloc"(%0) : (tensor<2x2xf32>) -> ()
    "polar_graph.lock"(%0) {lock_type = "read_write"} : (tensor<2x2xf32>) -> ()
    "polar_graph.invoke_loader"(%0) : (tensor<2x2xf32>) -> ()
    "polar_graph.release"(%0) : (tensor<2x2xf32>) -> ()
    polar_graph.return %0 : tensor<2x2xf32>
  }) {cluster_id = 0 : index, node_id = 4 : index, sym_name = "inp1", type = () -> tensor<2x2xf32>} : () -> ()
  "polar_graph.node"() ( {
    %0 = "polar_graph.create_tensor"() {virtual_address = 17 : index} : () -> tensor<2x2xf32>
    "polar_graph.alloc"(%0) : (tensor<2x2xf32>) -> ()
    "polar_graph.lock"(%0) {lock_type = "read_write"} : (tensor<2x2xf32>) -> ()
    "polar_graph.invoke_loader"(%0) : (tensor<2x2xf32>) -> ()
    "polar_graph.release"(%0) : (tensor<2x2xf32>) -> ()
    polar_graph.return %0 : tensor<2x2xf32>
  }) {cluster_id = 0 : index, node_id = 6 : index, sym_name = "inp2", type = () -> tensor<2x2xf32>} : () -> ()
  "polar_graph.node"() ( {
  ^bb0(%arg0: tensor<2x2xf32>, %arg1: tensor<2x2xf32>):  // no predecessors
    %0 = "polar_graph.create_tensor"() {virtual_address = 33 : index} : () -> tensor<2x2xf32>
    "polar_graph.alloc"(%0) : (tensor<2x2xf32>) -> ()
    %cst = constant 1.000000e+00 : f32
    %cst_0 = constant 1.000000e+00 : f32
    "polar_graph.lock"(%0) {lock_type = "read_write"} : (tensor<2x2xf32>) -> ()
    "polar_graph.lock"(%arg1) {lock_type = "read"} : (tensor<2x2xf32>) -> ()
    "polar_graph.lock"(%arg0) {lock_type = "read"} : (tensor<2x2xf32>) -> ()
    "polar_graph.add"(%arg0, %cst, %arg1, %cst_0, %0) : (tensor<2x2xf32>, f32, tensor<2x2xf32>, f32, tensor<2x2xf32>) -> ()
    "polar_graph.release"(%arg1) : (tensor<2x2xf32>) -> ()
    "polar_graph.release"(%arg0) : (tensor<2x2xf32>) -> ()
    "polar_graph.release"(%0) : (tensor<2x2xf32>) -> ()
    polar_graph.return %0 : tensor<2x2xf32>
  }) {cluster_id = 1 : index, node_id = 9 : index, sym_name = "add", type = (tensor<2x2xf32>, tensor<2x2xf32>) -> tensor<2x2xf32>} : () -> ()
  "polar_graph.graph"() ( {
    %0 = polar_graph.eval @inp1() : () -> tensor<2x2xf32>
    %1 = polar_graph.eval @inp2() : () -> tensor<2x2xf32>
    "polar_graph.barrier"() {clusterId = 0 : index} : () -> ()
    %2 = polar_graph.eval @add(%0, %1) : (tensor<2x2xf32>, tensor<2x2xf32>) -> tensor<2x2xf32>
    "polar_graph.barrier"() {clusterId = 1 : index} : () -> ()
    "polar_graph.barrier"() {clusterId = 2 : index} : () -> ()
    "polar_graph.graph_terminator"() : () -> ()
  }) {sym_name = "mainGraph", type = () -> ()} : () -> ()
}

// CHECK: module {
// CHECK-NEXT: func @inputA(%arg0: !polar_rt.graph_handle) -> !polar_rt.event attributes {cluster_id = 0 : index, node_id = 0 : index} {
// CHECK-NEXT: %0 = "polar_graph.create_tensor"() {virtual_address = 1 : index} : () -> tensor<8xf32>
// CHECK-NEXT: "polar_graph.alloc"(%0) : (tensor<8xf32>) -> ()
// CHECK-NEXT: "polar_graph.lock"(%0) {lock_type = "read_write"} : (tensor<8xf32>) -> ()
// CHECK-NEXT: "polar_graph.invoke_loader"(%0) : (tensor<8xf32>) -> ()
// CHECK-NEXT: "polar_graph.release"(%0) : (tensor<8xf32>) -> ()
// CHECK-NEXT: %1 = "polar_rt.null_event"() : () -> !polar_rt.event
// CHECK-NEXT: return %1 : !polar_rt.event
// CHECK-NEXT: }
// CHECK-NEXT: func @inputB(%arg0: !polar_rt.graph_handle) -> !polar_rt.event attributes {cluster_id = 0 : index, node_id = 1 : index} {
// CHECK-NEXT: %0 = "polar_graph.create_tensor"() {virtual_address = 9 : index} : () -> tensor<8xf32>
// CHECK-NEXT: "polar_graph.alloc"(%0) : (tensor<8xf32>) -> ()
// CHECK-NEXT: "polar_graph.lock"(%0) {lock_type = "read_write"} : (tensor<8xf32>) -> ()
// CHECK-NEXT: "polar_graph.invoke_loader"(%0) : (tensor<8xf32>) -> ()
// CHECK-NEXT: "polar_graph.release"(%0) : (tensor<8xf32>) -> ()
// CHECK-NEXT: %1 = "polar_rt.null_event"() : () -> !polar_rt.event
// CHECK-NEXT: return %1 : !polar_rt.event
// CHECK-NEXT: }
// CHECK-NEXT: func @sum(%arg0: !polar_rt.graph_handle) -> !polar_rt.event attributes {cluster_id = 1 : index, node_id = 2 : index} {
// CHECK-NEXT: %0 = "polar_graph.create_tensor"() {virtual_address = 9 : index} : () -> tensor<8xf32>
// CHECK-NEXT: %1 = "polar_graph.create_tensor"() {virtual_address = 1 : index} : () -> tensor<8xf32>
// CHECK-NEXT: %2 = "polar_graph.create_tensor"() {virtual_address = 17 : index} : () -> tensor<8xf32>
// CHECK-NEXT: "polar_graph.lock"(%1) {lock_type = "read"} : (tensor<8xf32>) -> ()
// CHECK-NEXT: "polar_graph.lock"(%0) {lock_type = "read"} : (tensor<8xf32>) -> ()
// CHECK-NEXT: "polar_graph.alloc"(%2) : (tensor<8xf32>) -> ()
// CHECK-NEXT: "polar_graph.lock"(%2) {lock_type = "read_write"} : (tensor<8xf32>) -> ()
// CHECK-NEXT: %cst = constant 1.000000e+00 : f32
// CHECK-NEXT: %3 = "polar_rt.select_device"() {nodeId = 2 : index} : () -> !polar_rt.device
// CHECK-NEXT: %4 = "polar_rt.null_event"() : () -> !polar_rt.event
// CHECK-NEXT: %5 = "polar_rt.launch"(%3, %4, %1, %cst, %0, %cst, %2) {global_size = [8], kernel = "dummy", local_size = [0]} : (!polar_rt.device, !polar_rt.event, tensor<8xf32>, f32, tensor<8xf32>, f32, tensor<8xf32>) -> (!polar_rt.event)
// CHECK-NEXT: "polar_graph.release"(%1) : (tensor<8xf32>) -> ()
// CHECK-NEXT: "polar_graph.release"(%0) : (tensor<8xf32>) -> ()
// CHECK-NEXT: "polar_graph.release"(%2) : (tensor<8xf32>) -> ()
// CHECK-NEXT: return %5 : !polar_rt.event
// CHECK-NEXT: }
// CHECK-NEXT: func @mainGraph(%arg0: !polar_rt.graph_handle) {
// CHECK-NEXT: %0 = call @inputA(%arg0) : (!polar_rt.graph_handle) -> !polar_rt.event
// CHECK-NEXT: %1 = call @inputB(%arg0) : (!polar_rt.graph_handle) -> !polar_rt.event
// CHECK-NEXT: "polar_rt.barrier"() {cluster_id = 0 : index} : () -> ()
// CHECK-NEXT: %2 = call @sum(%arg0) : (!polar_rt.graph_handle) -> !polar_rt.event
// CHECK-NEXT: return
// CHECK-NEXT: }
// CHECK-NEXT: }
