// RUN: athena-opt --convert-runtime-to-llvm %s | FileCheck %s
module {
  llvm.func @ath_allocate(!llvm<"i8*">, !llvm<"i8*">, !llvm<"i8*">)
  llvm.func @ath_release(!llvm<"i8*">, !llvm<"i8*">, !llvm<"i8*">)
  llvm.func @ath_lock(!llvm<"i8*">, !llvm<"i8*">, !llvm<"i8*">, !llvm.i32)
  llvm.func @ath_device_select(!llvm<"i8*">, !llvm.i64) -> !llvm<"i8*">
  llvm.func @ath_load(!llvm<"i8*">, !llvm.i64, !llvm<"i8*">)
  llvm.func @ath_barrier(!llvm.i64, !llvm<"i8*">)
  llvm.func @ath_launch(!llvm<"i8*">, !llvm<"i8*">, !llvm<"i8*">, !llvm<"{ i8*, i32, { i64, i8*, i32 }*, i64, i64*, i64* }*">) -> !llvm<"i8*">
  func @inputA(%arg0: !ath_rt.graph_handle) -> !ath_rt.event attributes {cluster_id = 0 : index, node_id = 0 : index} {
    %0 = "ath_graph.create_tensor"() {virtual_address = 1 : index} : () -> tensor<8xf32>
    %1 = "ath_rt.select_device"() {nodeId = 0 : index} : () -> !ath_rt.device
    "ath_rt.alloc"(%1, %0) : (!ath_rt.device, tensor<8xf32>) -> ()
    %2 = "ath_rt.select_device"() {nodeId = 0 : index} : () -> !ath_rt.device
    "ath_rt.lock"(%2, %0) {lock_type = "read_write"} : (!ath_rt.device, tensor<8xf32>) -> ()
    "ath_graph.invoke_loader"(%0) {node_id = 0 : i64} : (tensor<8xf32>) -> ()
    %3 = "ath_rt.select_device"() {nodeId = 0 : index} : () -> !ath_rt.device
    "ath_rt.release"(%3, %0) : (!ath_rt.device, tensor<8xf32>) -> ()
    %4 = "ath_rt.null_event"() : () -> !ath_rt.event
    return %4 : !ath_rt.event
  }
  func @inputB(%arg0: !ath_rt.graph_handle) -> !ath_rt.event attributes {cluster_id = 0 : index, node_id = 1 : index} {
    %0 = "ath_graph.create_tensor"() {virtual_address = 9 : index} : () -> tensor<8xf32>
    %1 = "ath_rt.select_device"() {nodeId = 1 : index} : () -> !ath_rt.device
    "ath_rt.alloc"(%1, %0) : (!ath_rt.device, tensor<8xf32>) -> ()
    %2 = "ath_rt.select_device"() {nodeId = 1 : index} : () -> !ath_rt.device
    "ath_rt.lock"(%2, %0) {lock_type = "read_write"} : (!ath_rt.device, tensor<8xf32>) -> ()
    "ath_graph.invoke_loader"(%0) {node_id = 0 : i64} : (tensor<8xf32>) -> ()
    %3 = "ath_rt.select_device"() {nodeId = 1 : index} : () -> !ath_rt.device
    "ath_rt.release"(%3, %0) : (!ath_rt.device, tensor<8xf32>) -> ()
    %4 = "ath_rt.null_event"() : () -> !ath_rt.event
    return %4 : !ath_rt.event
  }
  func @sum(%arg0: !ath_rt.graph_handle) -> !ath_rt.event attributes {cluster_id = 1 : index, node_id = 2 : index} {
    %0 = "ath_graph.create_tensor"() {virtual_address = 9 : index} : () -> tensor<8xf32>
    %1 = "ath_graph.create_tensor"() {virtual_address = 1 : index} : () -> tensor<8xf32>
    %2 = "ath_graph.create_tensor"() {virtual_address = 17 : index} : () -> tensor<8xf32>
    %3 = "ath_rt.select_device"() {nodeId = 2 : index} : () -> !ath_rt.device
    "ath_rt.lock"(%3, %1) {lock_type = "read"} : (!ath_rt.device, tensor<8xf32>) -> ()
    %4 = "ath_rt.select_device"() {nodeId = 2 : index} : () -> !ath_rt.device
    "ath_rt.lock"(%4, %0) {lock_type = "read"} : (!ath_rt.device, tensor<8xf32>) -> ()
    %5 = "ath_rt.select_device"() {nodeId = 2 : index} : () -> !ath_rt.device
    "ath_rt.alloc"(%5, %2) : (!ath_rt.device, tensor<8xf32>) -> ()
    %6 = "ath_rt.select_device"() {nodeId = 2 : index} : () -> !ath_rt.device
    "ath_rt.lock"(%6, %2) {lock_type = "read_write"} : (!ath_rt.device, tensor<8xf32>) -> ()
    %cst = constant 1.000000e+00 : f32
    %7 = "ath_rt.select_device"() {nodeId = 2 : index} : () -> !ath_rt.device
    %8 = "ath_rt.null_event"() : () -> !ath_rt.event
    %out_tensor, %out_event = "ath_rt.launch"(%7, %8, %1, %cst, %0, %cst, %2) {global_size = [8], kernel = "dummy", local_size = [0]} : (!ath_rt.device, !ath_rt.event, tensor<8xf32>, f32, tensor<8xf32>, f32, tensor<8xf32>) -> (tensor<8xf32>, !ath_rt.event)
    %9 = "ath_rt.select_device"() {nodeId = 2 : index} : () -> !ath_rt.device
    "ath_rt.release"(%9, %1) : (!ath_rt.device, tensor<8xf32>) -> ()
    %10 = "ath_rt.select_device"() {nodeId = 2 : index} : () -> !ath_rt.device
    "ath_rt.release"(%10, %0) : (!ath_rt.device, tensor<8xf32>) -> ()
    %11 = "ath_rt.select_device"() {nodeId = 2 : index} : () -> !ath_rt.device
    "ath_rt.release"(%11, %2) : (!ath_rt.device, tensor<8xf32>) -> ()
    return %out_event : !ath_rt.event
  }
  func @mainGraph(%arg0: !ath_rt.graph_handle) {
    %0 = call @inputA(%arg0) : (!ath_rt.graph_handle) -> !ath_rt.event
    %1 = call @inputB(%arg0) : (!ath_rt.graph_handle) -> !ath_rt.event
    "ath_rt.barrier"(%0, %1) : (!ath_rt.event, !ath_rt.event) -> ()
    %2 = call @sum(%arg0) : (!ath_rt.graph_handle) -> !ath_rt.event
    return
  }
}

// CHECK: module {
// CHECK-NEXT: llvm.mlir.global private @dummy("dummy")
// CHECK-NEXT: llvm.func @ath_allocate(!llvm<"i8*">, !llvm<"i8*">, !llvm<"i8*">)
// CHECK-NEXT: llvm.func @ath_release(!llvm<"i8*">, !llvm<"i8*">, !llvm<"i8*">)
// CHECK-NEXT: llvm.func @ath_lock(!llvm<"i8*">, !llvm<"i8*">, !llvm<"i8*">, !llvm.i32)
// CHECK-NEXT: llvm.func @ath_device_select(!llvm<"i8*">, !llvm.i64) -> !llvm<"i8*">
// CHECK-NEXT: llvm.func @ath_load(!llvm<"i8*">, !llvm.i64, !llvm<"i8*">)
// CHECK-NEXT: llvm.func @ath_barrier(!llvm.i64, !llvm<"i8*">)
// CHECK-NEXT: llvm.func @ath_launch(!llvm<"i8*">, !llvm<"i8*">, !llvm<"i8*">, !llvm<"{ i8*, i32, { i64, i8*, i32 }*, i64, i64*, i64* }*">) -> !llvm<"i8*">
// CHECK-NEXT: llvm.func @inputA(%arg0: !llvm<"i8*">) -> !llvm<"i8*"> attributes {cluster_id = 0 : index, node_id = 0 : index} {
// CHECK-NEXT: %0 = llvm.mlir.constant(1 : i32) : !llvm.i32
// CHECK-NEXT: %1 = llvm.alloca %0 x !llvm<"{ i64, i32, i64, i64* }"> {alignment = 8 : i64} : (!llvm.i32) -> !llvm<"{ i64, i32, i64, i64* }*">
// CHECK-NEXT: %2 = llvm.mlir.constant(1 : i64) : !llvm.i64
// CHECK-NEXT: %3 = llvm.mlir.constant(0 : i32) : !llvm.i32
// CHECK-NEXT: %4 = llvm.mlir.constant(0 : i32) : !llvm.i32
// CHECK-NEXT: %5 = llvm.getelementptr %1[%3, %4] : (!llvm<"{ i64, i32, i64, i64* }*">, !llvm.i32, !llvm.i32) -> !llvm<"i64*">
// CHECK-NEXT: llvm.store %2, %5 : !llvm<"i64*">
// CHECK-NEXT: %6 = llvm.mlir.constant(2 : i32) : !llvm.i32
// CHECK-NEXT: %7 = llvm.mlir.constant(0 : i32) : !llvm.i32
// CHECK-NEXT: %8 = llvm.mlir.constant(1 : i32) : !llvm.i32
// CHECK-NEXT: %9 = llvm.getelementptr %1[%7, %8] : (!llvm<"{ i64, i32, i64, i64* }*">, !llvm.i32, !llvm.i32) -> !llvm<"i32*">
// CHECK-NEXT: llvm.store %6, %9 : !llvm<"i32*">
// CHECK-NEXT: %10 = llvm.mlir.constant(1 : i64) : !llvm.i64
// CHECK-NEXT: %11 = llvm.mlir.constant(0 : i32) : !llvm.i32
// CHECK-NEXT: %12 = llvm.mlir.constant(2 : i32) : !llvm.i32
// CHECK-NEXT: %13 = llvm.getelementptr %1[%11, %12] : (!llvm<"{ i64, i32, i64, i64* }*">, !llvm.i32, !llvm.i32) -> !llvm<"i64*">
// CHECK-NEXT: llvm.store %10, %13 : !llvm<"i64*">
// CHECK-NEXT: %14 = llvm.mlir.constant(1 : i32) : !llvm.i32
// CHECK-NEXT: %15 = llvm.alloca %14 x !llvm.i32 {alignment = 16 : i64} : (!llvm.i32) -> !llvm<"i32*">
// CHECK-NEXT: %16 = llvm.mlir.constant(8 : i64) : !llvm.i64
// CHECK-NEXT: %17 = llvm.mlir.constant(0 : i32) : !llvm.i32
// CHECK-NEXT: %18 = llvm.mlir.constant(0 : i32) : !llvm.i32
// CHECK-NEXT: %19 = llvm.getelementptr %15[%17, %18] : (!llvm<"i32*">, !llvm.i32, !llvm.i32) -> !llvm.i32
// CHECK-NEXT: llvm.store %16, %19 : !llvm.i32
// CHECK-NEXT: %20 = llvm.mlir.constant(0 : i32) : !llvm.i32
// CHECK-NEXT: %21 = llvm.mlir.constant(3 : i32) : !llvm.i32
// CHECK-NEXT: %22 = llvm.getelementptr %1[%20, %21] : (!llvm<"{ i64, i32, i64, i64* }*">, !llvm.i32, !llvm.i32) -> !llvm<"i64**">
// CHECK-NEXT: llvm.store %15, %22 : !llvm<"i64**">
// CHECK-NEXT: %23 = llvm.mlir.constant(0 : i64) : !llvm.i64
// CHECK-NEXT: %24 = llvm.call @ath_device_select(%arg0, %23) : (!llvm<"i8*">, !llvm.i64) -> !llvm<"i8*">
// CHECK-NEXT: llvm.call @ath_allocate(%arg0, %24, %1) : (!llvm<"i8*">, !llvm<"i8*">, !llvm<"{ i64, i32, i64, i64* }*">) -> ()
// CHECK-NEXT: %25 = llvm.mlir.constant(0 : i64) : !llvm.i64
// CHECK-NEXT: %26 = llvm.call @ath_device_select(%arg0, %25) : (!llvm<"i8*">, !llvm.i64) -> !llvm<"i8*">
// CHECK-NEXT: %27 = llvm.mlir.constant(1 : i32) : !llvm.i32
// CHECK-NEXT: llvm.call @ath_lock(%arg0, %26, %1, %27) : (!llvm<"i8*">, !llvm<"i8*">, !llvm<"{ i64, i32, i64, i64* }*">, !llvm.i32) -> ()
// CHECK-NEXT: %28 = llvm.mlir.constant(0 : i64) : !llvm.i64
// CHECK-NEXT: llvm.call @ath_load(%arg0, %28, %1) : (!llvm<"i8*">, !llvm.i64, !llvm<"{ i64, i32, i64, i64* }*">) -> ()
// CHECK-NEXT: %29 = llvm.mlir.constant(0 : i64) : !llvm.i64
// CHECK-NEXT: %30 = llvm.call @ath_device_select(%arg0, %29) : (!llvm<"i8*">, !llvm.i64) -> !llvm<"i8*">
// CHECK-NEXT: llvm.call @ath_release(%arg0, %30, %1) : (!llvm<"i8*">, !llvm<"i8*">, !llvm<"{ i64, i32, i64, i64* }*">) -> ()
// CHECK-NEXT: %31 = llvm.mlir.null : !llvm<"i8*">
// CHECK-NEXT: llvm.return %31 : !llvm<"i8*">
// CHECK-NEXT: }
// CHECK-NEXT: llvm.func @inputB(%arg0: !llvm<"i8*">) -> !llvm<"i8*"> attributes {cluster_id = 0 : index, node_id = 1 : index} {
// CHECK-NEXT: %0 = llvm.mlir.constant(1 : i32) : !llvm.i32
// CHECK-NEXT: %1 = llvm.alloca %0 x !llvm<"{ i64, i32, i64, i64* }"> {alignment = 8 : i64} : (!llvm.i32) -> !llvm<"{ i64, i32, i64, i64* }*">
// CHECK-NEXT: %2 = llvm.mlir.constant(9 : i64) : !llvm.i64
// CHECK-NEXT: %3 = llvm.mlir.constant(0 : i32) : !llvm.i32
// CHECK-NEXT: %4 = llvm.mlir.constant(0 : i32) : !llvm.i32
// CHECK-NEXT: %5 = llvm.getelementptr %1[%3, %4] : (!llvm<"{ i64, i32, i64, i64* }*">, !llvm.i32, !llvm.i32) -> !llvm<"i64*">
// CHECK-NEXT: llvm.store %2, %5 : !llvm<"i64*">
// CHECK-NEXT: %6 = llvm.mlir.constant(2 : i32) : !llvm.i32
// CHECK-NEXT: %7 = llvm.mlir.constant(0 : i32) : !llvm.i32
// CHECK-NEXT: %8 = llvm.mlir.constant(1 : i32) : !llvm.i32
// CHECK-NEXT: %9 = llvm.getelementptr %1[%7, %8] : (!llvm<"{ i64, i32, i64, i64* }*">, !llvm.i32, !llvm.i32) -> !llvm<"i32*">
// CHECK-NEXT: llvm.store %6, %9 : !llvm<"i32*">
// CHECK-NEXT: %10 = llvm.mlir.constant(1 : i64) : !llvm.i64
// CHECK-NEXT: %11 = llvm.mlir.constant(0 : i32) : !llvm.i32
// CHECK-NEXT: %12 = llvm.mlir.constant(2 : i32) : !llvm.i32
// CHECK-NEXT: %13 = llvm.getelementptr %1[%11, %12] : (!llvm<"{ i64, i32, i64, i64* }*">, !llvm.i32, !llvm.i32) -> !llvm<"i64*">
// CHECK-NEXT: llvm.store %10, %13 : !llvm<"i64*">
// CHECK-NEXT: %14 = llvm.mlir.constant(1 : i32) : !llvm.i32
// CHECK-NEXT: %15 = llvm.alloca %14 x !llvm.i32 {alignment = 16 : i64} : (!llvm.i32) -> !llvm<"i32*">
// CHECK-NEXT: %16 = llvm.mlir.constant(8 : i64) : !llvm.i64
// CHECK-NEXT: %17 = llvm.mlir.constant(0 : i32) : !llvm.i32
// CHECK-NEXT: %18 = llvm.mlir.constant(0 : i32) : !llvm.i32
// CHECK-NEXT: %19 = llvm.getelementptr %15[%17, %18] : (!llvm<"i32*">, !llvm.i32, !llvm.i32) -> !llvm.i32
// CHECK-NEXT: llvm.store %16, %19 : !llvm.i32
// CHECK-NEXT: %20 = llvm.mlir.constant(0 : i32) : !llvm.i32
// CHECK-NEXT: %21 = llvm.mlir.constant(3 : i32) : !llvm.i32
// CHECK-NEXT: %22 = llvm.getelementptr %1[%20, %21] : (!llvm<"{ i64, i32, i64, i64* }*">, !llvm.i32, !llvm.i32) -> !llvm<"i64**">
// CHECK-NEXT: llvm.store %15, %22 : !llvm<"i64**">
// CHECK-NEXT: %23 = llvm.mlir.constant(1 : i64) : !llvm.i64
// CHECK-NEXT: %24 = llvm.call @ath_device_select(%arg0, %23) : (!llvm<"i8*">, !llvm.i64) -> !llvm<"i8*">
// CHECK-NEXT: llvm.call @ath_allocate(%arg0, %24, %1) : (!llvm<"i8*">, !llvm<"i8*">, !llvm<"{ i64, i32, i64, i64* }*">) -> ()
// CHECK-NEXT: %25 = llvm.mlir.constant(1 : i64) : !llvm.i64
// CHECK-NEXT: %26 = llvm.call @ath_device_select(%arg0, %25) : (!llvm<"i8*">, !llvm.i64) -> !llvm<"i8*">
// CHECK-NEXT: %27 = llvm.mlir.constant(1 : i32) : !llvm.i32
// CHECK-NEXT: llvm.call @ath_lock(%arg0, %26, %1, %27) : (!llvm<"i8*">, !llvm<"i8*">, !llvm<"{ i64, i32, i64, i64* }*">, !llvm.i32) -> ()
// CHECK-NEXT: %28 = llvm.mlir.constant(0 : i64) : !llvm.i64
// CHECK-NEXT: llvm.call @ath_load(%arg0, %28, %1) : (!llvm<"i8*">, !llvm.i64, !llvm<"{ i64, i32, i64, i64* }*">) -> ()
// CHECK-NEXT: %29 = llvm.mlir.constant(1 : i64) : !llvm.i64
// CHECK-NEXT: %30 = llvm.call @ath_device_select(%arg0, %29) : (!llvm<"i8*">, !llvm.i64) -> !llvm<"i8*">
// CHECK-NEXT: llvm.call @ath_release(%arg0, %30, %1) : (!llvm<"i8*">, !llvm<"i8*">, !llvm<"{ i64, i32, i64, i64* }*">) -> ()
// CHECK-NEXT: %31 = llvm.mlir.null : !llvm<"i8*">
// CHECK-NEXT: llvm.return %31 : !llvm<"i8*">
// CHECK-NEXT: }
// CHECK-NEXT: llvm.func @sum(%arg0: !llvm<"i8*">) -> !llvm<"i8*"> attributes {cluster_id = 1 : index, node_id = 2 : index} {
// CHECK-NEXT: %0 = llvm.mlir.constant(1 : i32) : !llvm.i32
// CHECK-NEXT: %1 = llvm.alloca %0 x !llvm<"{ i64, i32, i64, i64* }"> {alignment = 8 : i64} : (!llvm.i32) -> !llvm<"{ i64, i32, i64, i64* }*">
// CHECK-NEXT: %2 = llvm.mlir.constant(9 : i64) : !llvm.i64
// CHECK-NEXT: %3 = llvm.mlir.constant(0 : i32) : !llvm.i32
// CHECK-NEXT: %4 = llvm.mlir.constant(0 : i32) : !llvm.i32
// CHECK-NEXT: %5 = llvm.getelementptr %1[%3, %4] : (!llvm<"{ i64, i32, i64, i64* }*">, !llvm.i32, !llvm.i32) -> !llvm<"i64*">
// CHECK-NEXT: llvm.store %2, %5 : !llvm<"i64*">
// CHECK-NEXT: %6 = llvm.mlir.constant(2 : i32) : !llvm.i32
// CHECK-NEXT: %7 = llvm.mlir.constant(0 : i32) : !llvm.i32
// CHECK-NEXT: %8 = llvm.mlir.constant(1 : i32) : !llvm.i32
// CHECK-NEXT: %9 = llvm.getelementptr %1[%7, %8] : (!llvm<"{ i64, i32, i64, i64* }*">, !llvm.i32, !llvm.i32) -> !llvm<"i32*">
// CHECK-NEXT: llvm.store %6, %9 : !llvm<"i32*">
// CHECK-NEXT: %10 = llvm.mlir.constant(1 : i64) : !llvm.i64
// CHECK-NEXT: %11 = llvm.mlir.constant(0 : i32) : !llvm.i32
// CHECK-NEXT: %12 = llvm.mlir.constant(2 : i32) : !llvm.i32
// CHECK-NEXT: %13 = llvm.getelementptr %1[%11, %12] : (!llvm<"{ i64, i32, i64, i64* }*">, !llvm.i32, !llvm.i32) -> !llvm<"i64*">
// CHECK-NEXT: llvm.store %10, %13 : !llvm<"i64*">
// CHECK-NEXT: %14 = llvm.mlir.constant(1 : i32) : !llvm.i32
// CHECK-NEXT: %15 = llvm.alloca %14 x !llvm.i32 {alignment = 16 : i64} : (!llvm.i32) -> !llvm<"i32*">
// CHECK-NEXT: %16 = llvm.mlir.constant(8 : i64) : !llvm.i64
// CHECK-NEXT: %17 = llvm.mlir.constant(0 : i32) : !llvm.i32
// CHECK-NEXT: %18 = llvm.mlir.constant(0 : i32) : !llvm.i32
// CHECK-NEXT: %19 = llvm.getelementptr %15[%17, %18] : (!llvm<"i32*">, !llvm.i32, !llvm.i32) -> !llvm.i32
// CHECK-NEXT: llvm.store %16, %19 : !llvm.i32
// CHECK-NEXT: %20 = llvm.mlir.constant(0 : i32) : !llvm.i32
// CHECK-NEXT: %21 = llvm.mlir.constant(3 : i32) : !llvm.i32
// CHECK-NEXT: %22 = llvm.getelementptr %1[%20, %21] : (!llvm<"{ i64, i32, i64, i64* }*">, !llvm.i32, !llvm.i32) -> !llvm<"i64**">
// CHECK-NEXT: llvm.store %15, %22 : !llvm<"i64**">
// CHECK-NEXT: %23 = llvm.mlir.constant(1 : i32) : !llvm.i32
// CHECK-NEXT: %24 = llvm.alloca %23 x !llvm<"{ i64, i32, i64, i64* }"> {alignment = 8 : i64} : (!llvm.i32) -> !llvm<"{ i64, i32, i64, i64* }*">
// CHECK-NEXT: %25 = llvm.mlir.constant(1 : i64) : !llvm.i64
// CHECK-NEXT: %26 = llvm.mlir.constant(0 : i32) : !llvm.i32
// CHECK-NEXT: %27 = llvm.mlir.constant(0 : i32) : !llvm.i32
// CHECK-NEXT: %28 = llvm.getelementptr %24[%26, %27] : (!llvm<"{ i64, i32, i64, i64* }*">, !llvm.i32, !llvm.i32) -> !llvm<"i64*">
// CHECK-NEXT: llvm.store %25, %28 : !llvm<"i64*">
// CHECK-NEXT: %29 = llvm.mlir.constant(2 : i32) : !llvm.i32
// CHECK-NEXT: %30 = llvm.mlir.constant(0 : i32) : !llvm.i32
// CHECK-NEXT: %31 = llvm.mlir.constant(1 : i32) : !llvm.i32
// CHECK-NEXT: %32 = llvm.getelementptr %24[%30, %31] : (!llvm<"{ i64, i32, i64, i64* }*">, !llvm.i32, !llvm.i32) -> !llvm<"i32*">
// CHECK-NEXT: llvm.store %29, %32 : !llvm<"i32*">
// CHECK-NEXT: %33 = llvm.mlir.constant(1 : i64) : !llvm.i64
// CHECK-NEXT: %34 = llvm.mlir.constant(0 : i32) : !llvm.i32
// CHECK-NEXT: %35 = llvm.mlir.constant(2 : i32) : !llvm.i32
// CHECK-NEXT: %36 = llvm.getelementptr %24[%34, %35] : (!llvm<"{ i64, i32, i64, i64* }*">, !llvm.i32, !llvm.i32) -> !llvm<"i64*">
// CHECK-NEXT: llvm.store %33, %36 : !llvm<"i64*">
// CHECK-NEXT: %37 = llvm.mlir.constant(1 : i32) : !llvm.i32
// CHECK-NEXT: %38 = llvm.alloca %37 x !llvm.i32 {alignment = 16 : i64} : (!llvm.i32) -> !llvm<"i32*">
// CHECK-NEXT: %39 = llvm.mlir.constant(8 : i64) : !llvm.i64
// CHECK-NEXT: %40 = llvm.mlir.constant(0 : i32) : !llvm.i32
// CHECK-NEXT: %41 = llvm.mlir.constant(0 : i32) : !llvm.i32
// CHECK-NEXT: %42 = llvm.getelementptr %38[%40, %41] : (!llvm<"i32*">, !llvm.i32, !llvm.i32) -> !llvm.i32
// CHECK-NEXT: llvm.store %39, %42 : !llvm.i32
// CHECK-NEXT: %43 = llvm.mlir.constant(0 : i32) : !llvm.i32
// CHECK-NEXT: %44 = llvm.mlir.constant(3 : i32) : !llvm.i32
// CHECK-NEXT: %45 = llvm.getelementptr %24[%43, %44] : (!llvm<"{ i64, i32, i64, i64* }*">, !llvm.i32, !llvm.i32) -> !llvm<"i64**">
// CHECK-NEXT: llvm.store %38, %45 : !llvm<"i64**">
// CHECK-NEXT: %46 = llvm.mlir.constant(1 : i32) : !llvm.i32
// CHECK-NEXT: %47 = llvm.alloca %46 x !llvm<"{ i64, i32, i64, i64* }"> {alignment = 8 : i64} : (!llvm.i32) -> !llvm<"{ i64, i32, i64, i64* }*">
// CHECK-NEXT: %48 = llvm.mlir.constant(17 : i64) : !llvm.i64
// CHECK-NEXT: %49 = llvm.mlir.constant(0 : i32) : !llvm.i32
// CHECK-NEXT: %50 = llvm.mlir.constant(0 : i32) : !llvm.i32
// CHECK-NEXT: %51 = llvm.getelementptr %47[%49, %50] : (!llvm<"{ i64, i32, i64, i64* }*">, !llvm.i32, !llvm.i32) -> !llvm<"i64*">
// CHECK-NEXT: llvm.store %48, %51 : !llvm<"i64*">
// CHECK-NEXT: %52 = llvm.mlir.constant(2 : i32) : !llvm.i32
// CHECK-NEXT: %53 = llvm.mlir.constant(0 : i32) : !llvm.i32
// CHECK-NEXT: %54 = llvm.mlir.constant(1 : i32) : !llvm.i32
// CHECK-NEXT: %55 = llvm.getelementptr %47[%53, %54] : (!llvm<"{ i64, i32, i64, i64* }*">, !llvm.i32, !llvm.i32) -> !llvm<"i32*">
// CHECK-NEXT: llvm.store %52, %55 : !llvm<"i32*">
// CHECK-NEXT: %56 = llvm.mlir.constant(1 : i64) : !llvm.i64
// CHECK-NEXT: %57 = llvm.mlir.constant(0 : i32) : !llvm.i32
// CHECK-NEXT: %58 = llvm.mlir.constant(2 : i32) : !llvm.i32
// CHECK-NEXT: %59 = llvm.getelementptr %47[%57, %58] : (!llvm<"{ i64, i32, i64, i64* }*">, !llvm.i32, !llvm.i32) -> !llvm<"i64*">
// CHECK-NEXT: llvm.store %56, %59 : !llvm<"i64*">
// CHECK-NEXT: %60 = llvm.mlir.constant(1 : i32) : !llvm.i32
// CHECK-NEXT: %61 = llvm.alloca %60 x !llvm.i32 {alignment = 16 : i64} : (!llvm.i32) -> !llvm<"i32*">
// CHECK-NEXT: %62 = llvm.mlir.constant(8 : i64) : !llvm.i64
// CHECK-NEXT: %63 = llvm.mlir.constant(0 : i32) : !llvm.i32
// CHECK-NEXT: %64 = llvm.mlir.constant(0 : i32) : !llvm.i32
// CHECK-NEXT: %65 = llvm.getelementptr %61[%63, %64] : (!llvm<"i32*">, !llvm.i32, !llvm.i32) -> !llvm.i32
// CHECK-NEXT: llvm.store %62, %65 : !llvm.i32
// CHECK-NEXT: %66 = llvm.mlir.constant(0 : i32) : !llvm.i32
// CHECK-NEXT: %67 = llvm.mlir.constant(3 : i32) : !llvm.i32
// CHECK-NEXT: %68 = llvm.getelementptr %47[%66, %67] : (!llvm<"{ i64, i32, i64, i64* }*">, !llvm.i32, !llvm.i32) -> !llvm<"i64**">
// CHECK-NEXT: llvm.store %61, %68 : !llvm<"i64**">
// CHECK-NEXT: %69 = llvm.mlir.constant(2 : i64) : !llvm.i64
// CHECK-NEXT: %70 = llvm.call @ath_device_select(%arg0, %69) : (!llvm<"i8*">, !llvm.i64) -> !llvm<"i8*">
// CHECK-NEXT: %71 = llvm.mlir.constant(0 : i32) : !llvm.i32
// CHECK-NEXT: llvm.call @ath_lock(%arg0, %70, %24, %71) : (!llvm<"i8*">, !llvm<"i8*">, !llvm<"{ i64, i32, i64, i64* }*">, !llvm.i32) -> ()
// CHECK-NEXT: %72 = llvm.mlir.constant(2 : i64) : !llvm.i64
// CHECK-NEXT: %73 = llvm.call @ath_device_select(%arg0, %72) : (!llvm<"i8*">, !llvm.i64) -> !llvm<"i8*">
// CHECK-NEXT: %74 = llvm.mlir.constant(0 : i32) : !llvm.i32
// CHECK-NEXT: llvm.call @ath_lock(%arg0, %73, %1, %74) : (!llvm<"i8*">, !llvm<"i8*">, !llvm<"{ i64, i32, i64, i64* }*">, !llvm.i32) -> ()
// CHECK-NEXT: %75 = llvm.mlir.constant(2 : i64) : !llvm.i64
// CHECK-NEXT: %76 = llvm.call @ath_device_select(%arg0, %75) : (!llvm<"i8*">, !llvm.i64) -> !llvm<"i8*">
// CHECK-NEXT: llvm.call @ath_allocate(%arg0, %76, %47) : (!llvm<"i8*">, !llvm<"i8*">, !llvm<"{ i64, i32, i64, i64* }*">) -> ()
// CHECK-NEXT: %77 = llvm.mlir.constant(2 : i64) : !llvm.i64
// CHECK-NEXT: %78 = llvm.call @ath_device_select(%arg0, %77) : (!llvm<"i8*">, !llvm.i64) -> !llvm<"i8*">
// CHECK-NEXT: %79 = llvm.mlir.constant(1 : i32) : !llvm.i32
// CHECK-NEXT: llvm.call @ath_lock(%arg0, %78, %47, %79) : (!llvm<"i8*">, !llvm<"i8*">, !llvm<"{ i64, i32, i64, i64* }*">, !llvm.i32) -> ()
// CHECK-NEXT: %80 = llvm.mlir.constant(1.000000e+00 : f32) : !llvm.float
// CHECK-NEXT: %81 = llvm.mlir.constant(2 : i64) : !llvm.i64
// CHECK-NEXT: %82 = llvm.call @ath_device_select(%arg0, %81) : (!llvm<"i8*">, !llvm.i64) -> !llvm<"i8*">
// CHECK-NEXT: %83 = llvm.mlir.null : !llvm<"i8*">
// CHECK-NEXT: %84 = llvm.mlir.constant(4 : i32) : !llvm.i32
// CHECK-NEXT: %85 = llvm.alloca %84 x !llvm<"{ i64, i8*, i32 }"> {alignment = 16 : i64} : (!llvm.i32) -> !llvm<"{ i64, i8*, i32 }*">
// CHECK-NEXT: %86 = llvm.mlir.constant(1 : i32) : !llvm.i32
// CHECK-NEXT: %87 = llvm.alloca %86 x !llvm<"{ i64, i8*, i32 }"> {alignment = 8 : i64} : (!llvm.i32) -> !llvm<"{ i64, i8*, i32 }*">
// CHECK-NEXT: %88 = llvm.mlir.constant(4 : i64) : !llvm.i64
// CHECK-NEXT: %89 = llvm.mlir.constant(0 : i32) : !llvm.i32
// CHECK-NEXT: %90 = llvm.mlir.constant(0 : i32) : !llvm.i32
// CHECK-NEXT: %91 = llvm.getelementptr %87[%89, %90] : (!llvm<"{ i64, i8*, i32 }*">, !llvm.i32, !llvm.i32) -> !llvm<"i64*">
// CHECK-NEXT: llvm.store %88, %91 : !llvm<"i64*">
// CHECK-NEXT: %92 = llvm.mlir.constant(1 : i64) : !llvm.i64
// CHECK-NEXT: %93 = llvm.alloca %92 x !llvm.float {alignment = 8 : i64} : (!llvm.i64) -> !llvm<"float*">
// CHECK-NEXT: llvm.store %80, %93 : !llvm<"float*">
// CHECK-NEXT: %94 = llvm.bitcast %93 : !llvm<"float*"> to !llvm<"i8*">
// CHECK-NEXT: %95 = llvm.mlir.constant(0 : i32) : !llvm.i32
// CHECK-NEXT: %96 = llvm.mlir.constant(1 : i32) : !llvm.i32
// CHECK-NEXT: %97 = llvm.getelementptr %87[%95, %96] : (!llvm<"{ i64, i8*, i32 }*">, !llvm.i32, !llvm.i32) -> !llvm<"i8**">
// CHECK-NEXT: llvm.store %94, %97 : !llvm<"i8**">
// CHECK-NEXT: %98 = llvm.mlir.constant(0 : i32) : !llvm.i32
// CHECK-NEXT: %99 = llvm.mlir.constant(2 : i32) : !llvm.i32
// CHECK-NEXT: %100 = llvm.getelementptr %87[%98, %99] : (!llvm<"{ i64, i8*, i32 }*">, !llvm.i32, !llvm.i32) -> !llvm<"i32*">
// CHECK-NEXT: llvm.store %92, %100 : !llvm<"i32*">
// CHECK-NEXT: %101 = llvm.mlir.constant(0 : i32) : !llvm.i32
// CHECK-NEXT: %102 = llvm.mlir.constant(0 : i32) : !llvm.i32
// CHECK-NEXT: %103 = llvm.getelementptr %85[%101, %102] : (!llvm<"{ i64, i8*, i32 }*">, !llvm.i32, !llvm.i32) -> !llvm<"{ i64, i8*, i32 }">
// CHECK-NEXT: llvm.store %87, %103 : !llvm<"{ i64, i8*, i32 }">
// CHECK-NEXT: %104 = llvm.mlir.constant(1 : i32) : !llvm.i32
// CHECK-NEXT: %105 = llvm.alloca %104 x !llvm<"{ i64, i8*, i32 }"> {alignment = 8 : i64} : (!llvm.i32) -> !llvm<"{ i64, i8*, i32 }*">
// CHECK-NEXT: %106 = llvm.mlir.constant(0 : i64) : !llvm.i64
// CHECK-NEXT: %107 = llvm.mlir.constant(0 : i32) : !llvm.i32
// CHECK-NEXT: %108 = llvm.mlir.constant(0 : i32) : !llvm.i32
// CHECK-NEXT: %109 = llvm.getelementptr %105[%107, %108] : (!llvm<"{ i64, i8*, i32 }*">, !llvm.i32, !llvm.i32) -> !llvm<"i64*">
// CHECK-NEXT: llvm.store %106, %109 : !llvm<"i64*">
// CHECK-NEXT: %110 = llvm.mlir.constant(1 : i64) : !llvm.i64
// CHECK-NEXT: %111 = llvm.alloca %110 x !llvm<"{ i64, i32, i64, i64* }*"> {alignment = 8 : i64} : (!llvm.i64) -> !llvm<"{ i64, i32, i64, i64* }**">
// CHECK-NEXT: llvm.store %1, %111 : !llvm<"{ i64, i32, i64, i64* }**">
// CHECK-NEXT: %112 = llvm.bitcast %111 : !llvm<"{ i64, i32, i64, i64* }**"> to !llvm<"i8*">
// CHECK-NEXT: %113 = llvm.mlir.constant(0 : i32) : !llvm.i32
// CHECK-NEXT: %114 = llvm.mlir.constant(1 : i32) : !llvm.i32
// CHECK-NEXT: %115 = llvm.getelementptr %105[%113, %114] : (!llvm<"{ i64, i8*, i32 }*">, !llvm.i32, !llvm.i32) -> !llvm<"i8**">
// CHECK-NEXT: llvm.store %112, %115 : !llvm<"i8**">
// CHECK-NEXT: %116 = llvm.mlir.constant(0 : i32) : !llvm.i32
// CHECK-NEXT: %117 = llvm.mlir.constant(2 : i32) : !llvm.i32
// CHECK-NEXT: %118 = llvm.getelementptr %105[%116, %117] : (!llvm<"{ i64, i8*, i32 }*">, !llvm.i32, !llvm.i32) -> !llvm<"i32*">
// CHECK-NEXT: llvm.store %110, %118 : !llvm<"i32*">
// CHECK-NEXT: %119 = llvm.mlir.constant(0 : i32) : !llvm.i32
// CHECK-NEXT: %120 = llvm.mlir.constant(1 : i32) : !llvm.i32
// CHECK-NEXT: %121 = llvm.getelementptr %85[%119, %120] : (!llvm<"{ i64, i8*, i32 }*">, !llvm.i32, !llvm.i32) -> !llvm<"{ i64, i8*, i32 }">
// CHECK-NEXT: llvm.store %105, %121 : !llvm<"{ i64, i8*, i32 }">
// CHECK-NEXT: %122 = llvm.mlir.constant(1 : i32) : !llvm.i32
// CHECK-NEXT: %123 = llvm.alloca %122 x !llvm<"{ i64, i8*, i32 }"> {alignment = 8 : i64} : (!llvm.i32) -> !llvm<"{ i64, i8*, i32 }*">
// CHECK-NEXT: %124 = llvm.mlir.constant(4 : i64) : !llvm.i64
// CHECK-NEXT: %125 = llvm.mlir.constant(0 : i32) : !llvm.i32
// CHECK-NEXT: %126 = llvm.mlir.constant(0 : i32) : !llvm.i32
// CHECK-NEXT: %127 = llvm.getelementptr %123[%125, %126] : (!llvm<"{ i64, i8*, i32 }*">, !llvm.i32, !llvm.i32) -> !llvm<"i64*">
// CHECK-NEXT: llvm.store %124, %127 : !llvm<"i64*">
// CHECK-NEXT: %128 = llvm.mlir.constant(1 : i64) : !llvm.i64
// CHECK-NEXT: %129 = llvm.alloca %128 x !llvm.float {alignment = 8 : i64} : (!llvm.i64) -> !llvm<"float*">
// CHECK-NEXT: llvm.store %80, %129 : !llvm<"float*">
// CHECK-NEXT: %130 = llvm.bitcast %129 : !llvm<"float*"> to !llvm<"i8*">
// CHECK-NEXT: %131 = llvm.mlir.constant(0 : i32) : !llvm.i32
// CHECK-NEXT: %132 = llvm.mlir.constant(1 : i32) : !llvm.i32
// CHECK-NEXT: %133 = llvm.getelementptr %123[%131, %132] : (!llvm<"{ i64, i8*, i32 }*">, !llvm.i32, !llvm.i32) -> !llvm<"i8**">
// CHECK-NEXT: llvm.store %130, %133 : !llvm<"i8**">
// CHECK-NEXT: %134 = llvm.mlir.constant(0 : i32) : !llvm.i32
// CHECK-NEXT: %135 = llvm.mlir.constant(2 : i32) : !llvm.i32
// CHECK-NEXT: %136 = llvm.getelementptr %123[%134, %135] : (!llvm<"{ i64, i8*, i32 }*">, !llvm.i32, !llvm.i32) -> !llvm<"i32*">
// CHECK-NEXT: llvm.store %128, %136 : !llvm<"i32*">
// CHECK-NEXT: %137 = llvm.mlir.constant(0 : i32) : !llvm.i32
// CHECK-NEXT: %138 = llvm.mlir.constant(2 : i32) : !llvm.i32
// CHECK-NEXT: %139 = llvm.getelementptr %85[%137, %138] : (!llvm<"{ i64, i8*, i32 }*">, !llvm.i32, !llvm.i32) -> !llvm<"{ i64, i8*, i32 }">
// CHECK-NEXT: llvm.store %123, %139 : !llvm<"{ i64, i8*, i32 }">
// CHECK-NEXT: %140 = llvm.mlir.constant(1 : i32) : !llvm.i32
// CHECK-NEXT: %141 = llvm.alloca %140 x !llvm<"{ i64, i8*, i32 }"> {alignment = 8 : i64} : (!llvm.i32) -> !llvm<"{ i64, i8*, i32 }*">
// CHECK-NEXT: %142 = llvm.mlir.constant(0 : i64) : !llvm.i64
// CHECK-NEXT: %143 = llvm.mlir.constant(0 : i32) : !llvm.i32
// CHECK-NEXT: %144 = llvm.mlir.constant(0 : i32) : !llvm.i32
// CHECK-NEXT: %145 = llvm.getelementptr %141[%143, %144] : (!llvm<"{ i64, i8*, i32 }*">, !llvm.i32, !llvm.i32) -> !llvm<"i64*">
// CHECK-NEXT: llvm.store %142, %145 : !llvm<"i64*">
// CHECK-NEXT: %146 = llvm.mlir.constant(1 : i64) : !llvm.i64
// CHECK-NEXT: %147 = llvm.alloca %146 x !llvm<"{ i64, i32, i64, i64* }*"> {alignment = 8 : i64} : (!llvm.i64) -> !llvm<"{ i64, i32, i64, i64* }**">
// CHECK-NEXT: llvm.store %47, %147 : !llvm<"{ i64, i32, i64, i64* }**">
// CHECK-NEXT: %148 = llvm.bitcast %147 : !llvm<"{ i64, i32, i64, i64* }**"> to !llvm<"i8*">
// CHECK-NEXT: %149 = llvm.mlir.constant(0 : i32) : !llvm.i32
// CHECK-NEXT: %150 = llvm.mlir.constant(1 : i32) : !llvm.i32
// CHECK-NEXT: %151 = llvm.getelementptr %141[%149, %150] : (!llvm<"{ i64, i8*, i32 }*">, !llvm.i32, !llvm.i32) -> !llvm<"i8**">
// CHECK-NEXT: llvm.store %148, %151 : !llvm<"i8**">
// CHECK-NEXT: %152 = llvm.mlir.constant(0 : i32) : !llvm.i32
// CHECK-NEXT: %153 = llvm.mlir.constant(2 : i32) : !llvm.i32
// CHECK-NEXT: %154 = llvm.getelementptr %141[%152, %153] : (!llvm<"{ i64, i8*, i32 }*">, !llvm.i32, !llvm.i32) -> !llvm<"i32*">
// CHECK-NEXT: llvm.store %146, %154 : !llvm<"i32*">
// CHECK-NEXT: %155 = llvm.mlir.constant(0 : i32) : !llvm.i32
// CHECK-NEXT: %156 = llvm.mlir.constant(3 : i32) : !llvm.i32
// CHECK-NEXT: %157 = llvm.getelementptr %85[%155, %156] : (!llvm<"{ i64, i8*, i32 }*">, !llvm.i32, !llvm.i32) -> !llvm<"{ i64, i8*, i32 }">
// CHECK-NEXT: llvm.store %141, %157 : !llvm<"{ i64, i8*, i32 }">
// CHECK-NEXT: %158 = llvm.mlir.constant(1 : i32) : !llvm.i32
// CHECK-NEXT: %159 = llvm.alloca %158 x !llvm<"{ i8*, i32, { i64, i8*, i32 }*, i64, i64*, i64* }"> {alignment = 8 : i64} : (!llvm.i32) -> !llvm<"{ i8*, i32, { i64, i8*, i32 }*, i64, i64*, i64* }*">
// CHECK-NEXT: %160 = llvm.mlir.addressof @dummy : !llvm<"[5 x i8]*">
// CHECK-NEXT: %161 = llvm.mlir.constant(0 : i32) : !llvm.i32
// CHECK-NEXT: %162 = llvm.mlir.constant(0 : i32) : !llvm.i32
// CHECK-NEXT: %163 = llvm.getelementptr %159[%161, %162] : (!llvm<"{ i8*, i32, { i64, i8*, i32 }*, i64, i64*, i64* }*">, !llvm.i32, !llvm.i32) -> !llvm<"i8**">
// CHECK-NEXT: llvm.store %160, %163 : !llvm<"i8**">
// CHECK-NEXT: %164 = llvm.mlir.constant(4 : i64) : !llvm.i64
// CHECK-NEXT: %165 = llvm.mlir.constant(0 : i32) : !llvm.i32
// CHECK-NEXT: %166 = llvm.mlir.constant(1 : i32) : !llvm.i32
// CHECK-NEXT: %167 = llvm.getelementptr %159[%165, %166] : (!llvm<"{ i8*, i32, { i64, i8*, i32 }*, i64, i64*, i64* }*">, !llvm.i32, !llvm.i32) -> !llvm<"i32*">
// CHECK-NEXT: llvm.store %164, %167 : !llvm<"i32*">
// CHECK-NEXT: %168 = llvm.mlir.constant(0 : i32) : !llvm.i32
// CHECK-NEXT: %169 = llvm.mlir.constant(2 : i32) : !llvm.i32
// CHECK-NEXT: %170 = llvm.getelementptr %159[%168, %169] : (!llvm<"{ i8*, i32, { i64, i8*, i32 }*, i64, i64*, i64* }*">, !llvm.i32, !llvm.i32) -> !llvm<"{ i64, i8*, i32 }**">
// CHECK-NEXT: llvm.store %85, %170 : !llvm<"{ i64, i8*, i32 }**">
// CHECK-NEXT: %171 = llvm.mlir.constant(1 : i64) : !llvm.i64
// CHECK-NEXT: %172 = llvm.mlir.constant(0 : i32) : !llvm.i32
// CHECK-NEXT: %173 = llvm.mlir.constant(3 : i32) : !llvm.i32
// CHECK-NEXT: %174 = llvm.getelementptr %159[%172, %173] : (!llvm<"{ i8*, i32, { i64, i8*, i32 }*, i64, i64*, i64* }*">, !llvm.i32, !llvm.i32) -> !llvm<"i64*">
// CHECK-NEXT: llvm.store %171, %174 : !llvm<"i64*">
// CHECK-NEXT: %175 = llvm.mlir.constant(1 : i32) : !llvm.i32
// CHECK-NEXT: %176 = llvm.alloca %175 x !llvm.i64 {alignment = 16 : i64} : (!llvm.i32) -> !llvm<"i64*">
// CHECK-NEXT: %177 = llvm.mlir.constant(8 : i64) : !llvm.i64
// CHECK-NEXT: %178 = llvm.mlir.constant(0 : i32) : !llvm.i32
// CHECK-NEXT: %179 = llvm.mlir.constant(0 : i32) : !llvm.i32
// CHECK-NEXT: %180 = llvm.getelementptr %176[%178, %179] : (!llvm<"i64*">, !llvm.i32, !llvm.i32) -> !llvm.i64
// CHECK-NEXT: llvm.store %177, %180 : !llvm.i64
// CHECK-NEXT: %181 = llvm.mlir.constant(0 : i32) : !llvm.i32
// CHECK-NEXT: %182 = llvm.mlir.constant(4 : i32) : !llvm.i32
// CHECK-NEXT: %183 = llvm.getelementptr %159[%181, %182] : (!llvm<"{ i8*, i32, { i64, i8*, i32 }*, i64, i64*, i64* }*">, !llvm.i32, !llvm.i32) -> !llvm<"i64**">
// CHECK-NEXT: llvm.store %176, %183 : !llvm<"i64**">
// CHECK-NEXT: %184 = llvm.mlir.constant(1 : i32) : !llvm.i32
// CHECK-NEXT: %185 = llvm.alloca %184 x !llvm.i64 {alignment = 16 : i64} : (!llvm.i32) -> !llvm<"i64*">
// CHECK-NEXT: %186 = llvm.mlir.constant(0 : i64) : !llvm.i64
// CHECK-NEXT: %187 = llvm.mlir.constant(0 : i32) : !llvm.i32
// CHECK-NEXT: %188 = llvm.mlir.constant(0 : i32) : !llvm.i32
// CHECK-NEXT: %189 = llvm.getelementptr %185[%187, %188] : (!llvm<"i64*">, !llvm.i32, !llvm.i32) -> !llvm.i64
// CHECK-NEXT: llvm.store %186, %189 : !llvm.i64
// CHECK-NEXT: %190 = llvm.mlir.constant(0 : i32) : !llvm.i32
// CHECK-NEXT: %191 = llvm.mlir.constant(5 : i32) : !llvm.i32
// CHECK-NEXT: %192 = llvm.getelementptr %159[%190, %191] : (!llvm<"{ i8*, i32, { i64, i8*, i32 }*, i64, i64*, i64* }*">, !llvm.i32, !llvm.i32) -> !llvm<"i64**">
// CHECK-NEXT: llvm.store %185, %192 : !llvm<"i64**">
// CHECK-NEXT: %193 = llvm.call @ath_launch(%arg0, %82, %83, %159) : (!llvm<"i8*">, !llvm<"i8*">, !llvm<"i8*">, !llvm<"{ i8*, i32, { i64, i8*, i32 }*, i64, i64*, i64* }*">) -> !llvm<"i8*">
// CHECK-NEXT: %194 = llvm.mlir.constant(2 : i64) : !llvm.i64
// CHECK-NEXT: %195 = llvm.call @ath_device_select(%arg0, %194) : (!llvm<"i8*">, !llvm.i64) -> !llvm<"i8*">
// CHECK-NEXT: llvm.call @ath_release(%arg0, %195, %24) : (!llvm<"i8*">, !llvm<"i8*">, !llvm<"{ i64, i32, i64, i64* }*">) -> ()
// CHECK-NEXT: %196 = llvm.mlir.constant(2 : i64) : !llvm.i64
// CHECK-NEXT: %197 = llvm.call @ath_device_select(%arg0, %196) : (!llvm<"i8*">, !llvm.i64) -> !llvm<"i8*">
// CHECK-NEXT: llvm.call @ath_release(%arg0, %197, %1) : (!llvm<"i8*">, !llvm<"i8*">, !llvm<"{ i64, i32, i64, i64* }*">) -> ()
// CHECK-NEXT: %198 = llvm.mlir.constant(2 : i64) : !llvm.i64
// CHECK-NEXT: %199 = llvm.call @ath_device_select(%arg0, %198) : (!llvm<"i8*">, !llvm.i64) -> !llvm<"i8*">
// CHECK-NEXT: llvm.call @ath_release(%arg0, %199, %47) : (!llvm<"i8*">, !llvm<"i8*">, !llvm<"{ i64, i32, i64, i64* }*">) -> ()
// CHECK-NEXT: llvm.return %193 : !llvm<"i8*">
// CHECK-NEXT: }
// CHECK-NEXT: llvm.func @mainGraph(%arg0: !llvm<"i8*">) {
// CHECK-NEXT: %0 = llvm.call @inputA(%arg0) : (!llvm<"i8*">) -> !llvm<"i8*">
// CHECK-NEXT: %1 = llvm.call @inputB(%arg0) : (!llvm<"i8*">) -> !llvm<"i8*">
// CHECK-NEXT: %2 = llvm.mlir.constant(2 : i32) : !llvm.i32
// CHECK-NEXT: %3 = llvm.mlir.constant(2 : i32) : !llvm.i32
// CHECK-NEXT: %4 = llvm.alloca %3 x !llvm<"i8*"> {alignment = 16 : i64} : (!llvm.i32) -> !llvm<"i8**">
// CHECK-NEXT: %5 = llvm.mlir.constant(0 : i32) : !llvm.i32
// CHECK-NEXT: %6 = llvm.mlir.constant(0 : i32) : !llvm.i32
// CHECK-NEXT: %7 = llvm.getelementptr %4[%5, %6] : (!llvm<"i8**">, !llvm.i32, !llvm.i32) -> !llvm<"i8*">
// CHECK-NEXT: llvm.store %0, %7 : !llvm<"i8*">
// CHECK-NEXT: %8 = llvm.mlir.constant(0 : i32) : !llvm.i32
// CHECK-NEXT: %9 = llvm.mlir.constant(1 : i32) : !llvm.i32
// CHECK-NEXT: %10 = llvm.getelementptr %4[%8, %9] : (!llvm<"i8**">, !llvm.i32, !llvm.i32) -> !llvm<"i8*">
// CHECK-NEXT: llvm.store %1, %10 : !llvm<"i8*">
// CHECK-NEXT: llvm.call @ath_barrier(%2, %4) : (!llvm.i32, !llvm<"i8**">) -> ()
// CHECK-NEXT: %11 = llvm.call @sum(%arg0) : (!llvm<"i8*">) -> !llvm<"i8*">
// CHECK-NEXT: llvm.return
// CHECK-NEXT: }
// CHECK-NEXT: }
