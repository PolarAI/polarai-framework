// RUN: polar-opt --rt-shape-inference %s | FileCheck %s

#map0 = affine_map<(d0, d1) -> (d0, d1)>
#map1 = affine_map<() -> (0)>
#map2 = affine_map<() -> (2)>


module {
  func @inp1(%arg0: !polar_rt.graph_handle) -> !polar_rt.event attributes {cluster_id = 0 : index, node_id = 4 : index} {
    %0 = "polar_graph.create_tensor"() {virtual_address = 1 : index} : () -> tensor<2x2xf32>
    %1 = "polar_rt.select_device"() {nodeId = 4 : index} : () -> !polar_rt.device
    "polar_rt.alloc"(%1, %0) : (!polar_rt.device, tensor<2x2xf32>) -> ()
    "polar_rt.lock"(%1, %0) {lock_type = "read_write"} : (!polar_rt.device, tensor<2x2xf32>) -> ()
    "polar_graph.invoke_loader"(%0) : (tensor<2x2xf32>) -> ()
    "polar_rt.release"(%1, %0) : (!polar_rt.device, tensor<2x2xf32>) -> ()
    %2 = "polar_rt.null_event"() : () -> !polar_rt.event
    return %2 : !polar_rt.event
  }
  func @inp2(%arg0: !polar_rt.graph_handle) -> !polar_rt.event attributes {cluster_id = 0 : index, node_id = 6 : index} {
    %0 = "polar_graph.create_tensor"() {virtual_address = 17 : index} : () -> tensor<2x2xf32>
    %1 = "polar_rt.select_device"() {nodeId = 6 : index} : () -> !polar_rt.device
    "polar_rt.alloc"(%1, %0) : (!polar_rt.device, tensor<2x2xf32>) -> ()
    "polar_rt.lock"(%1, %0) {lock_type = "read_write"} : (!polar_rt.device, tensor<2x2xf32>) -> ()
    "polar_graph.invoke_loader"(%0) : (tensor<2x2xf32>) -> ()
    "polar_rt.release"(%1, %0) : (!polar_rt.device, tensor<2x2xf32>) -> ()
    %2 = "polar_rt.null_event"() : () -> !polar_rt.event
    return %2 : !polar_rt.event
  }
  func @add(%arg0: !polar_rt.graph_handle) -> !polar_rt.event attributes {cluster_id = 1 : index, node_id = 9 : index} {
    %cst = constant 1.000000e+00 : f32
    %0 = "polar_graph.create_tensor"() {virtual_address = 17 : index} : () -> tensor<2x2xf32>
    %1 = "polar_graph.create_tensor"() {virtual_address = 1 : index} : () -> tensor<2x2xf32>
    %2 = "polar_graph.create_tensor"() {virtual_address = 33 : index} : () -> tensor<2x2xf32>
    %3 = "polar_rt.select_device"() {nodeId = 9 : index} : () -> !polar_rt.device
    "polar_rt.alloc"(%3, %2) : (!polar_rt.device, tensor<2x2xf32>) -> ()
    "polar_rt.lock"(%3, %2) {lock_type = "read_write"} : (!polar_rt.device, tensor<2x2xf32>) -> ()
    "polar_rt.lock"(%3, %0) {lock_type = "read"} : (!polar_rt.device, tensor<2x2xf32>) -> ()
    "polar_rt.lock"(%3, %1) {lock_type = "read"} : (!polar_rt.device, tensor<2x2xf32>) -> ()
    %4 = "polar_rt.null_event"() : () -> !polar_rt.event
    %5 = "polar_rt.apply"(%3, %4, %1, %cst, %0, %cst, %2) ( {
    // CHECK: ^bb0(%arg1: memref<2x2xf32>, %arg2: f32, %arg3: memref<2x2xf32>, %arg4: f32, %arg5: memref<2x2xf32>):  // no predecessors
    ^bb0(%arg1: memref<?x?xf32>, %arg2: f32, %arg3: memref<?x?xf32>, %arg4: f32, %arg5: memref<?x?xf32>):  // no predecessors
      affine.for %arg6 = 0 to 2 {
        affine.for %arg7 = 0 to 2 {
          // CHECK: %6 = affine.load %arg1[%arg6, %arg7] : memref<2x2xf32>
          %6 = affine.load %arg1[%arg6, %arg7] : memref<?x?xf32>
          // CHECK: %7 = affine.load %arg3[%arg6, %arg7] : memref<2x2xf32>
          %7 = affine.load %arg3[%arg6, %arg7] : memref<?x?xf32>
          %8 = mulf %6, %arg2 : f32
          %9 = mulf %7, %arg4 : f32
          %10 = addf %8, %9 : f32
          // CHECK: affine.store %10, %arg5[%arg6, %arg7] : memref<2x2xf32>
          affine.store %10, %arg5[%arg6, %arg7] : memref<?x?xf32>
        }
      }
      "polar_rt.terminator"() : () -> ()
    }) {kernel_name = "fadd"} : (!polar_rt.device, !polar_rt.event, tensor<2x2xf32>, f32, tensor<2x2xf32>, f32, tensor<2x2xf32>) -> !polar_rt.event
    "polar_rt.release"(%3, %0) : (!polar_rt.device, tensor<2x2xf32>) -> ()
    "polar_rt.release"(%3, %1) : (!polar_rt.device, tensor<2x2xf32>) -> ()
    "polar_rt.release"(%3, %2) : (!polar_rt.device, tensor<2x2xf32>) -> ()
    return %4 : !polar_rt.event
  }
  func @mainGraph(%arg0: !polar_rt.graph_handle) {
    %0 = call @inp1(%arg0) : (!polar_rt.graph_handle) -> !polar_rt.event
    %1 = call @inp2(%arg0) : (!polar_rt.graph_handle) -> !polar_rt.event
    "polar_rt.barrier"() {cluster_id = 0 : index} : () -> ()
    %2 = call @add(%arg0) : (!polar_rt.graph_handle) -> !polar_rt.event
    "polar_rt.barrier"() {cluster_id = 1 : index} : () -> ()
    "polar_rt.barrier"() {cluster_id = 2 : index} : () -> ()
    return
  }
}
