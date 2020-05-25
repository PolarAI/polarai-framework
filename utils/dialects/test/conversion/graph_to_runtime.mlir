// RUN: athena-opt --deploy-default-functions --convert-graph-to-runtime --canonicalize %s | FileCheck %s

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

// CHECK: module {
// CHECK-NEXT: func @inputA(%arg0: !ath_rt.graph_handle) -> !ath_rt.event attributes {cluster_id = 0 : index, node_id = 0 : index} {
// CHECK-NEXT: %0 = "ath_graph.create_tensor"() {virtual_address = 1 : index} : () -> tensor<8xf32>
// CHECK-NEXT: "ath_graph.alloc"(%0) : (tensor<8xf32>) -> ()
// CHECK-NEXT: "ath_graph.lock"(%0) {lock_type = "read_write"} : (tensor<8xf32>) -> ()
// CHECK-NEXT: "ath_graph.invoke_loader"(%0) : (tensor<8xf32>) -> ()
// CHECK-NEXT: "ath_graph.release"(%0) : (tensor<8xf32>) -> ()
// CHECK-NEXT: %1 = "ath_rt.null_event"() : () -> !ath_rt.event
// CHECK-NEXT: return %1 : !ath_rt.event
// CHECK-NEXT: }
// CHECK-NEXT: func @inputB(%arg0: !ath_rt.graph_handle) -> !ath_rt.event attributes {cluster_id = 0 : index, node_id = 1 : index} {
// CHECK-NEXT: %0 = "ath_graph.create_tensor"() {virtual_address = 9 : index} : () -> tensor<8xf32>
// CHECK-NEXT: "ath_graph.alloc"(%0) : (tensor<8xf32>) -> ()
// CHECK-NEXT: "ath_graph.lock"(%0) {lock_type = "read_write"} : (tensor<8xf32>) -> ()
// CHECK-NEXT: "ath_graph.invoke_loader"(%0) : (tensor<8xf32>) -> ()
// CHECK-NEXT: "ath_graph.release"(%0) : (tensor<8xf32>) -> ()
// CHECK-NEXT: %1 = "ath_rt.null_event"() : () -> !ath_rt.event
// CHECK-NEXT: return %1 : !ath_rt.event
// CHECK-NEXT: }
// CHECK-NEXT: func @sum(%arg0: !ath_rt.graph_handle) -> !ath_rt.event attributes {cluster_id = 1 : index, node_id = 2 : index} {
// CHECK-NEXT: %0 = "ath_graph.create_tensor"() {virtual_address = 9 : index} : () -> tensor<8xf32>
// CHECK-NEXT: %1 = "ath_graph.create_tensor"() {virtual_address = 1 : index} : () -> tensor<8xf32>
// CHECK-NEXT: %2 = "ath_graph.create_tensor"() {virtual_address = 17 : index} : () -> tensor<8xf32>
// CHECK-NEXT: "ath_graph.lock"(%1) {lock_type = "read"} : (tensor<8xf32>) -> ()
// CHECK-NEXT: "ath_graph.lock"(%0) {lock_type = "read"} : (tensor<8xf32>) -> ()
// CHECK-NEXT: "ath_graph.alloc"(%2) : (tensor<8xf32>) -> ()
// CHECK-NEXT: "ath_graph.lock"(%2) {lock_type = "read_write"} : (tensor<8xf32>) -> ()
// CHECK-NEXT: %cst = constant 1.000000e+00 : f32
// CHECK-NEXT: %3 = "ath_rt.select_device"() {nodeId = 2 : index} : () -> !ath_rt.device
// CHECK-NEXT: %4 = "ath_rt.null_event"() : () -> !ath_rt.event
// CHECK-NEXT: %out_tensor, %out_event = "ath_rt.launch"(%3, %4, %1, %cst, %0, %cst, %2) {global_size = [8], kernel = "dummy", local_size = [0]} : (!ath_rt.device, !ath_rt.event, tensor<8xf32>, f32, tensor<8xf32>, f32, tensor<8xf32>) -> (tensor<8xf32>, !ath_rt.event)
// CHECK-NEXT: "ath_graph.release"(%1) : (tensor<8xf32>) -> ()
// CHECK-NEXT: "ath_graph.release"(%0) : (tensor<8xf32>) -> ()
// CHECK-NEXT: "ath_graph.release"(%2) : (tensor<8xf32>) -> ()
// CHECK-NEXT: return %out_event : !ath_rt.event
// CHECK-NEXT: }
// CHECK-NEXT: func @mainGraph(%arg0: !ath_rt.graph_handle) {
// CHECK-NEXT: %0 = call @inputA(%arg0) : (!ath_rt.graph_handle) -> !ath_rt.event
// CHECK-NEXT: %1 = call @inputB(%arg0) : (!ath_rt.graph_handle) -> !ath_rt.event
// CHECK-NEXT: "ath_rt.barrier"() {cluster_id = 0 : index} : () -> ()
// CHECK-NEXT: %2 = call @sum(%arg0) : (!ath_rt.graph_handle) -> !ath_rt.event
// CHECK-NEXT: return
// CHECK-NEXT: }
// CHECK-NEXT: }
