// RUN: polar-opt --outline-kernels %s | FileCheck %s

module {
  "compute.module"() ( {
    // CHECK: "compute.func"() ( {
    // CHECK-NEXT: ^bb0(%arg0: memref<2x2xf32>, %arg1: f32, %arg2: memref<2x2xf32>, %arg3: f32, %arg4: memref<2x2xf32>):  // no predecessors
    // CHECK-NEXT: %0 = "compute.global_id"() {dim = 0 : index} : () -> index
    // CHECK-NEXT: %1 = "compute.global_id"() {dim = 1 : index} : () -> index
    // CHECK-NEXT: %2 = load %arg0[%0, %1] : memref<2x2xf32>
    // CHECK-NEXT: %3 = load %arg2[%0, %1] : memref<2x2xf32>
    // CHECK-NEXT: %4 = mulf %2, %arg1 : f32
    // CHECK-NEXT: %5 = mulf %3, %arg3 : f32
    // CHECK-NEXT: %6 = addf %4, %5 : f32
    // CHECK-NEXT: store %6, %arg4[%0, %1] : memref<2x2xf32>
    // CHECK-NEXT: "compute.return"() : () -> ()
    // CHECK-NEXT: }) {kernel = true, sym_name = "add_fadd", type = (memref<2x2xf32>, f32, memref<2x2xf32>, f32, memref<2x2xf32>) -> ()} : () -> ()
    compute.module_end
  }) {sym_name = "kernels"} : () -> ()
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
    %c0 = constant 0 : index
    %c2 = constant 2 : index
    %c1 = constant 1 : index
    %0 = "polar_graph.create_tensor"() {virtual_address = 17 : index} : () -> tensor<2x2xf32>
    %1 = "polar_graph.create_tensor"() {virtual_address = 1 : index} : () -> tensor<2x2xf32>
    %2 = "polar_graph.create_tensor"() {virtual_address = 33 : index} : () -> tensor<2x2xf32>
    %3 = "polar_rt.select_device"() {nodeId = 9 : index} : () -> !polar_rt.device
    "polar_rt.alloc"(%3, %2) : (!polar_rt.device, tensor<2x2xf32>) -> ()
    "polar_rt.lock"(%3, %2) {lock_type = "read_write"} : (!polar_rt.device, tensor<2x2xf32>) -> ()
    "polar_rt.lock"(%3, %0) {lock_type = "read"} : (!polar_rt.device, tensor<2x2xf32>) -> ()
    "polar_rt.lock"(%3, %1) {lock_type = "read"} : (!polar_rt.device, tensor<2x2xf32>) -> ()
    %4 = "polar_rt.null_event"() : () -> !polar_rt.event
    // CHECK: %5 = "polar_rt.launch_func"(%3, %4, %1, %cst, %0, %cst, %2) {global_offset = [0, 0], global_size = [2, 2], kernel = @add_fadd, local_size = [0, 0, 0], native_kernel = "fadd"} : (!polar_rt.device, !polar_rt.event, tensor<2x2xf32>, f32, tensor<2x2xf32>, f32, tensor<2x2xf32>) -> !polar_rt.event
    %5 = "polar_rt.launch"(%3, %4, %1, %cst, %0, %cst, %2) ( {
    ^bb0(%arg1: memref<2x2xf32>, %arg2: f32, %arg3: memref<2x2xf32>, %arg4: f32, %arg5: memref<2x2xf32>):  // no predecessors
      %6 = "compute.global_id"() {dim = 0 : index} : () -> index
      %7 = "compute.global_id"() {dim = 1 : index} : () -> index
      %8 = load %arg1[%6, %7] : memref<2x2xf32>
      %9 = load %arg3[%6, %7] : memref<2x2xf32>
      %10 = mulf %8, %arg2 : f32
      %11 = mulf %9, %arg4 : f32
      %12 = addf %10, %11 : f32
      store %12, %arg5[%6, %7] : memref<2x2xf32>
      "polar_rt.terminator"() : () -> ()
    }) {global_offset = [0, 0], global_size = [2, 2], kernel_name = "fadd", local_size = [0, 0, 0]} : (!polar_rt.device, !polar_rt.event, tensor<2x2xf32>, f32, tensor<2x2xf32>, f32, tensor<2x2xf32>) -> !polar_rt.event
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
