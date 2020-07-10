// RUN: polar-opt --destroy-graph-relations --convert-graph-to-runtime --canonicalize %s | FileCheck %s

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

// CHECK: #map0 = affine_map<(d0, d1) -> (d0, d1)>
// CHECK-NEXT: #map1 = affine_map<() -> (0)>
// CHECK-NEXT: #map2 = affine_map<() -> (2)>


// CHECK: module {
// CHECH-NEXT: func @inp1(%arg0: !polar_rt.graph_handle) -> !polar_rt.event attributes {cluster_id = 0 : index, node_id = 4 : index} {
// CHECH-NEXT: %0 = "polar_graph.create_tensor"() {virtual_address = 1 : index} : () -> tensor<2x2xf32>
// CHECH-NEXT: %1 = "polar_rt.select_device"() {nodeId = 4 : index} : () -> !polar_rt.device
// CHECH-NEXT: "polar_rt.alloc"(%1, %0) : (!polar_rt.device, tensor<2x2xf32>) -> ()
// CHECH-NEXT: %2 = "polar_rt.select_device"() {nodeId = 4 : index} : () -> !polar_rt.device
// CHECH-NEXT: "polar_rt.lock"(%2, %0) {lock_type = "read_write"} : (!polar_rt.device, tensor<2x2xf32>) -> ()
// CHECH-NEXT: "polar_graph.invoke_loader"(%0) : (tensor<2x2xf32>) -> ()
// CHECH-NEXT: %3 = "polar_rt.select_device"() {nodeId = 4 : index} : () -> !polar_rt.device
// CHECH-NEXT: "polar_rt.release"(%3, %0) : (!polar_rt.device, tensor<2x2xf32>) -> ()
// CHECH-NEXT: %4 = "polar_rt.null_event"() : () -> !polar_rt.event
// CHECH-NEXT: return %4 : !polar_rt.event
// CHECH-NEXT: }
// CHECH-NEXT: func @inp2(%arg0: !polar_rt.graph_handle) -> !polar_rt.event attributes {cluster_id = 0 : index, node_id = 6 : index} {
// CHECH-NEXT: %0 = "polar_graph.create_tensor"() {virtual_address = 17 : index} : () -> tensor<2x2xf32>
// CHECH-NEXT: %1 = "polar_rt.select_device"() {nodeId = 6 : index} : () -> !polar_rt.device
// CHECH-NEXT: "polar_rt.alloc"(%1, %0) : (!polar_rt.device, tensor<2x2xf32>) -> ()
// CHECH-NEXT: %2 = "polar_rt.select_device"() {nodeId = 6 : index} : () -> !polar_rt.device
// CHECH-NEXT: "polar_rt.lock"(%2, %0) {lock_type = "read_write"} : (!polar_rt.device, tensor<2x2xf32>) -> ()
// CHECH-NEXT: "polar_graph.invoke_loader"(%0) : (tensor<2x2xf32>) -> ()
// CHECH-NEXT: %3 = "polar_rt.select_device"() {nodeId = 6 : index} : () -> !polar_rt.device
// CHECH-NEXT: "polar_rt.release"(%3, %0) : (!polar_rt.device, tensor<2x2xf32>) -> ()
// CHECH-NEXT: %4 = "polar_rt.null_event"() : () -> !polar_rt.event
// CHECH-NEXT: return %4 : !polar_rt.event
// CHECH-NEXT: }
// CHECH-NEXT: func @add(%arg0: !polar_rt.graph_handle) -> !polar_rt.event attributes {cluster_id = 1 : index, node_id = 9 : index} {
// CHECH-NEXT: %0 = "polar_graph.create_tensor"() {virtual_address = 17 : index} : () -> tensor<2x2xf32>
// CHECH-NEXT: %1 = "polar_graph.create_tensor"() {virtual_address = 1 : index} : () -> tensor<2x2xf32>
// CHECH-NEXT: %2 = "polar_graph.create_tensor"() {virtual_address = 33 : index} : () -> tensor<2x2xf32>
// CHECH-NEXT: %3 = "polar_rt.select_device"() {nodeId = 9 : index} : () -> !polar_rt.device
// CHECH-NEXT: "polar_rt.alloc"(%3, %2) : (!polar_rt.device, tensor<2x2xf32>) -> ()
// CHECH-NEXT: %cst = constant 1.000000e+00 : f32
// CHECH-NEXT: %cst_0 = constant 1.000000e+00 : f32
// CHECH-NEXT: %4 = "polar_rt.select_device"() {nodeId = 9 : index} : () -> !polar_rt.device
// CHECH-NEXT: "polar_rt.lock"(%4, %2) {lock_type = "read_write"} : (!polar_rt.device, tensor<2x2xf32>) -> ()
// CHECH-NEXT: %5 = "polar_rt.select_device"() {nodeId = 9 : index} : () -> !polar_rt.device
// CHECH-NEXT: "polar_rt.lock"(%5, %0) {lock_type = "read"} : (!polar_rt.device, tensor<2x2xf32>) -> ()
// CHECH-NEXT: %6 = "polar_rt.select_device"() {nodeId = 9 : index} : () -> !polar_rt.device
// CHECH-NEXT: "polar_rt.lock"(%6, %1) {lock_type = "read"} : (!polar_rt.device, tensor<2x2xf32>) -> ()
// CHECH-NEXT: %7 = "polar_rt.select_device"() {nodeId = 9 : index} : () -> !polar_rt.device
// CHECH-NEXT: %8 = "polar_rt.null_event"() : () -> !polar_rt.event
// CHECH-NEXT: %9 = "polar_rt.apply"(%7, %8, %1, %cst, %0, %cst_0, %2) ( {
// CHECH-NEXT: ^bb0(%arg1: memref<?x?xf32>, %arg2: f32, %arg3: memref<?x?xf32>, %arg4: f32, %arg5: memref<?x?xf32>):  // no predecessors
// CHECH-NEXT: %c0 = constant 0 : index
// CHECH-NEXT: %c2 = constant 2 : index
// CHECH-NEXT: %c2_1 = constant 2 : index
// CHECH-NEXT: affine.for %arg6 = 0 to 2 {
// CHECH-NEXT: affine.for %arg7 = 0 to 2 {
// CHECH-NEXT: %14 = affine.load %arg1[%arg6, %arg7] : memref<?x?xf32>
// CHECH-NEXT: %15 = affine.load %arg3[%arg6, %arg7] : memref<?x?xf32>
// CHECH-NEXT: %16 = mulf %14, %arg2 : f32
// CHECH-NEXT: %17 = mulf %15, %arg4 : f32
// CHECH-NEXT: %18 = addf %16, %17 : f32
// CHECH-NEXT: affine.store %18, %arg5[%arg6, %arg7] : memref<?x?xf32>
// CHECH-NEXT: }
// CHECH-NEXT: }
// CHECH-NEXT: "polar_rt.terminator"() : () -> ()
// CHECH-NEXT: }) {kernel_name = "fadd"} : (!polar_rt.device, !polar_rt.event, tensor<2x2xf32>, f32, tensor<2x2xf32>, f32, tensor<2x2xf32>) -> !polar_rt.event
// CHECH-NEXT: %10 = "polar_rt.select_device"() {nodeId = 9 : index} : () -> !polar_rt.device
// CHECH-NEXT: "polar_rt.release"(%10, %0) : (!polar_rt.device, tensor<2x2xf32>) -> ()
// CHECH-NEXT: %11 = "polar_rt.select_device"() {nodeId = 9 : index} : () -> !polar_rt.device
// CHECH-NEXT: "polar_rt.release"(%11, %1) : (!polar_rt.device, tensor<2x2xf32>) -> ()
// CHECH-NEXT: %12 = "polar_rt.select_device"() {nodeId = 9 : index} : () -> !polar_rt.device
// CHECH-NEXT: "polar_rt.release"(%12, %2) : (!polar_rt.device, tensor<2x2xf32>) -> ()
// CHECH-NEXT: %13 = "polar_rt.null_event"() : () -> !polar_rt.event
// CHECH-NEXT: return %13 : !polar_rt.event
// CHECH-NEXT: }
// CHECH-NEXT: func @mainGraph(%arg0: !polar_rt.graph_handle) {
// CHECH-NEXT: %0 = call @inp1(%arg0) : (!polar_rt.graph_handle) -> !polar_rt.event
// CHECH-NEXT: %1 = call @inp2(%arg0) : (!polar_rt.graph_handle) -> !polar_rt.event
// CHECH-NEXT: "polar_rt.barrier"() {cluster_id = 0 : index} : () -> ()
// CHECH-NEXT: %2 = call @add(%arg0) : (!polar_rt.graph_handle) -> !polar_rt.event
// CHECH-NEXT: "polar_rt.barrier"() {cluster_id = 1 : index} : () -> ()
// CHECH-NEXT: "polar_rt.barrier"() {cluster_id = 2 : index} : () -> ()
// CHECH-NEXT: return
// CHECH-NEXT: }
// CHECH-NEXT: }
