// RUN: polar-opt --outline-kernels %s | FileCheck %s

module {
  gpu.module @kernels {
    // CHECK: gpu.func @add_fadd(%arg0: memref<2x2xf32>, %arg1: f32, %arg2: memref<2x2xf32>, %arg3: f32, %arg4: memref<2x2xf32>) kernel attributes {global_size = [2, 2, 1], local_size = [0, 0, 0], spv.entry_point_abi = {local_size = dense<1> : vector<3xi32>}} {
    // CHECK-NEXT: %0 = "gpu.block_dim"() {dimension = "x"} : () -> index
    // CHECK-NEXT: %1 = "gpu.block_id"() {dimension = "x"} : () -> index
    // CHECK-NEXT: %2 = "gpu.thread_id"() {dimension = "x"} : () -> index
    // CHECK-NEXT: %3 = muli %0, %1 : index
    // CHECK-NEXT: %4 = addi %3, %2 : index
    // CHECK-NEXT: %5 = "gpu.block_dim"() {dimension = "y"} : () -> index
    // CHECK-NEXT: %6 = "gpu.block_id"() {dimension = "y"} : () -> index
    // CHECK-NEXT: %7 = "gpu.thread_id"() {dimension = "y"} : () -> index
    // CHECK-NEXT: %8 = muli %5, %6 : index
    // CHECK-NEXT: %9 = addi %8, %7 : index
    // CHECK-NEXT: %10 = load %arg0[%4, %9] : memref<2x2xf32>
    // CHECK-NEXT: %11 = load %arg2[%4, %9] : memref<2x2xf32>
    // CHECK-NEXT: %12 = mulf %10, %arg1 : f32
    // CHECK-NEXT: %13 = mulf %11, %arg3 : f32
    // CHECK-NEXT: %14 = addf %12, %13 : f32
    // CHECK-NEXT: store %14, %arg4[%4, %9] : memref<2x2xf32>
    // CHECK-NEXT: gpu.return
    // CHECK-NEXT: }
  }
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
    // CHECK: %5 = "polar_rt.launch_func"(%3, %4, %1, %cst, %0, %cst, %2) {kernel = @add_fadd, native_kernel = "fadd"} : (!polar_rt.device, !polar_rt.event, tensor<2x2xf32>, f32, tensor<2x2xf32>, f32, tensor<2x2xf32>) -> !polar_rt.event
    %5 = "polar_rt.launch"(%3, %4, %1, %cst, %0, %cst, %2) ( {
    ^bb0(%arg1: memref<2x2xf32>, %arg2: f32, %arg3: memref<2x2xf32>, %arg4: f32, %arg5: memref<2x2xf32>):  // no predecessors
      %6 = "gpu.block_dim"() {dimension = "x"} : () -> index
      %7 = "gpu.block_id"() {dimension = "x"} : () -> index
      %8 = "gpu.thread_id"() {dimension = "x"} : () -> index
      %9 = muli %6, %7 : index
      %10 = addi %9, %8 : index
      %11 = "gpu.block_dim"() {dimension = "y"} : () -> index
      %12 = "gpu.block_id"() {dimension = "y"} : () -> index
      %13 = "gpu.thread_id"() {dimension = "y"} : () -> index
      %14 = muli %11, %12 : index
      %15 = addi %14, %13 : index
      %16 = load %arg1[%10, %15] : memref<2x2xf32>
      %17 = load %arg3[%10, %15] : memref<2x2xf32>
      %18 = mulf %16, %arg2 : f32
      %19 = mulf %17, %arg4 : f32
      %20 = addf %18, %19 : f32
      store %20, %arg5[%10, %15] : memref<2x2xf32>
      "polar_rt.terminator"() : () -> ()
    }) {global_offset = [0, 0], global_size = [2, 2, 1], kernel_name = "fadd", local_size = [0, 0, 0]} : (!polar_rt.device, !polar_rt.event, tensor<2x2xf32>, f32, tensor<2x2xf32>, f32, tensor<2x2xf32>) -> !polar_rt.event
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
