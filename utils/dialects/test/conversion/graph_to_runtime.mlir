// RUN: athena-opt --deploy-default-functions --convert-graph-to-runtime --canonicalize %s | FileCheck %s

module {
  "ath_graph.node"() ( {
  ^bb0(%arg0: index, %arg1: index):	// no predecessors
    %0 = "ath_graph.get_tensor"(%arg0) {virtual_address = 1 : index} : (index) -> tensor<1x8xf32>
    %1 = "ath_graph.slice"(%arg1, %0) : (index, tensor<1x8xf32>) -> tensor<8xf32>
    "ath_graph.alloc"(%1) : (tensor<8xf32>) -> ()
    "ath_graph.lock"(%1) {lock_type = "read_write"} : (tensor<8xf32>) -> ()
    "ath_graph.invoke_loader"(%1) {loader_routine = "MyLoaderLoad"} : (tensor<8xf32>) -> ()
    "ath_graph.release"(%1) : (tensor<8xf32>) -> ()
    ath_graph.return %0 : tensor<1x8xf32>
  }) {cluster_id = 0 : index, node_id = 0 : index, sym_name = "InputA", type = (index, index) -> tensor<1x8xf32>} : () -> ()
  "ath_graph.node"() ( {
  ^bb0(%arg0: index, %arg1: index):	// no predecessors
    %0 = "ath_graph.get_tensor"(%arg0) {virtual_address = 9 : index} : (index) -> tensor<1x8xf32>
    %1 = "ath_graph.slice"(%arg1, %0) : (index, tensor<1x8xf32>) -> tensor<8xf32>
    "ath_graph.alloc"(%1) : (tensor<8xf32>) -> ()
    "ath_graph.lock"(%1) {lock_type = "read_write"} : (tensor<8xf32>) -> ()
    "ath_graph.invoke_loader"(%1) {loader_routine = "MyLoaderLoad"} : (tensor<8xf32>) -> ()
    "ath_graph.release"(%1) : (tensor<8xf32>) -> ()
    ath_graph.return %0 : tensor<1x8xf32>
  }) {cluster_id = 0 : index, node_id = 1 : index, sym_name = "InputB", type = (index, index) -> tensor<8xf32>} : () -> ()
  "ath_graph.node"() ( {
  ^bb0(%arg0: tensor<1x8xf32>, %arg1: tensor<1x8xf32>, %arg2: index, %arg3: index):	// no predecessors
    %0 = "ath_graph.get_tensor"(%arg2) {virtual_address = 17 : index} : (index) -> tensor<1x8xf32>
    %1 = "ath_graph.slice"(%arg3, %0) : (index, tensor<1x8xf32>) -> tensor<8xf32>
    %2 = "ath_graph.slice"(%arg3, %arg0) : (index, tensor<1x8xf32>) -> tensor<8xf32>
    %3 = "ath_graph.slice"(%arg3, %arg1) : (index, tensor<1x8xf32>) -> tensor<8xf32>
    "ath_graph.alloc"(%1) : (tensor<8xf32>) -> ()
    "ath_graph.lock"(%2) {lock_type = "read"} : (tensor<8xf32>) -> ()
    "ath_graph.lock"(%3) {lock_type = "read"} : (tensor<8xf32>) -> ()
    "ath_graph.lock"(%1) {lock_type = "read_write"} : (tensor<8xf32>) -> ()
    %cst = constant 1.000000e+00 : f32
    %4 = "ath_graph.add"(%2, %cst, %3, %cst, %1) : (tensor<8xf32>, f32, tensor<8xf32>, f32, tensor<8xf32>) -> tensor<8xf32>
    "ath_graph.release"(%1) : (tensor<8xf32>) -> ()
    "ath_graph.release"(%2) : (tensor<8xf32>) -> ()
    "ath_graph.release"(%3) : (tensor<8xf32>) -> ()
    ath_graph.return %0 : tensor<1x8xf32>
  }) {cluster_id = 1 : index, node_id = 2 : index, sym_name = "SumNode", type = (tensor<1x8xf32>, tensor<1x8xf32>, index, index) -> tensor<1x8xf32>} : () -> ()
  "ath_graph.graph"() ( {
  ^bb0(%arg0: index, %arg1: index):	// no predecessors
    %0 = ath_graph.eval @InputA(%arg0, %arg1) : (index, index) -> tensor<1x8xf32>
    %1 = ath_graph.eval @InputB(%arg0, %arg1) : (index, index) -> tensor<1x8xf32>
    %2 = ath_graph.eval @SumNode(%0, %1, %arg0, %arg1) : (tensor<1x8xf32>, tensor<1x8xf32>, index, index) -> tensor<1x8xf32>
    "ath_graph.graph_terminator"() : () -> ()
  }) {sym_name = "SampleGraph", type = (index, index) -> ()} : () -> ()
}

// CHECK: module {
// CHECK-NEXT: llvm.mlir.global private @ath_graph.add("ath_graph.add")
// CHECK-NEXT: llvm.func @ath_allocate(!llvm<"i8*">, !llvm<"i8*">, !llvm<"i8*">)
// CHECK-NEXT: llvm.func @ath_release_tensor(!llvm<"i8*">, !llvm<"i8*">, !llvm<"i8*">)
// CHECK-NEXT: llvm.func @ath_get_tensor_ptr(!llvm<"i8*">, !llvm<"i8*">, !llvm.i64) -> !llvm<"i8*">
// CHECK-NEXT: llvm.func @ath_lock_tensor(!llvm<"i8*">, !llvm<"i8*">, !llvm<"i8*">, !llvm.i32)
// CHECK-NEXT: llvm.func @ath_get_sub_tensor(!llvm.i64, !llvm<"i8*">, !llvm.i32) -> !llvm<"i8*">
// CHECK-NEXT: llvm.func @ath_get_device_for_node(!llvm.i64, !llvm<"i8*">, !llvm.i32) -> !llvm<"i8*">
// CHECK-NEXT: llvm.func @MyLoaderLoad(!llvm<"i8*">, !llvm<"i8*">, !llvm<"i8*">)
// CHECK-NEXT: llvm.func @ath_launch(!llvm<"i8*">, !llvm<"i8*">, !llvm<"{ i8*, i32, { i64, i8*, i32 }*, i64, i64*, i64* }">)
// CHECK-NEXT: llvm.func @InputA(%arg0: !llvm<"i8*">, %arg1: !llvm.i64, %arg2: !llvm<"i8*">, %arg3: !llvm<"i8*">) -> !llvm.i8 attributes {cluster_id = 0 : index, node_id = 0 : index} {
// CHECK-NEXT: %0 = llvm.call @ath_get_tensor_ptr(%arg0) : (!llvm<"i8*">) -> !llvm<"i8*">
// CHECK-NEXT: %1 = llvm.call @ath_get_sub_tensor(%arg1, %0) : (!llvm.i64, !llvm<"i8*">) -> !llvm<"i8*">
// CHECK-NEXT: llvm.call @ath_allocate(%1) : (!llvm<"i8*">) -> ()
// CHECK-NEXT: llvm.call @ath_lock_tensor(%1) : (!llvm<"i8*">) -> ()
// CHECK-NEXT: llvm.call @MyLoaderLoad(%arg2, %arg3, %1) : (!llvm<"i8*">, !llvm<"i8*">, !llvm<"i8*">) -> ()
// CHECK-NEXT: llvm.call @ath_release_tensor(%1) : (!llvm<"i8*">) -> ()
// CHECK-NEXT: llvm.return %0 : !llvm<"i8*">
// CHECK-NEXT: }
// CHECK-NEXT: llvm.func @InputB(%arg0: !llvm<"i8*">, %arg1: !llvm.i64, %arg2: !llvm<"i8*">, %arg3: !llvm<"i8*">) -> !llvm.i8 attributes {cluster_id = 0 : index, node_id = 1 : index} {
// CHECK-NEXT: %0 = llvm.call @ath_get_tensor_ptr(%arg0) : (!llvm<"i8*">) -> !llvm<"i8*">
// CHECK-NEXT: %1 = llvm.call @ath_get_sub_tensor(%arg1, %0) : (!llvm.i64, !llvm<"i8*">) -> !llvm<"i8*">
// CHECK-NEXT: llvm.call @ath_allocate(%1) : (!llvm<"i8*">) -> ()
// CHECK-NEXT: llvm.call @ath_lock_tensor(%1) : (!llvm<"i8*">) -> ()
// CHECK-NEXT: llvm.call @MyLoaderLoad(%arg2, %arg3, %1) : (!llvm<"i8*">, !llvm<"i8*">, !llvm<"i8*">) -> ()
// CHECK-NEXT: llvm.call @ath_release_tensor(%1) : (!llvm<"i8*">) -> ()
// CHECK-NEXT: llvm.return %0 : !llvm<"i8*">
// CHECK-NEXT: }
// CHECK-NEXT: llvm.func @SumNode(%arg0: !llvm<"i8*">, %arg1: !llvm<"i8*">, %arg2: !llvm<"i8*">, %arg3: !llvm.i64, %arg4: !llvm<"i8*">, %arg5: !llvm<"i8*">) -> !llvm.i8 attributes {cluster_id = 1 : index, node_id = 2 : index} {
// CHECK-NEXT: %0 = llvm.call @ath_get_tensor_ptr(%arg2) : (!llvm<"i8*">) -> !llvm<"i8*">
// CHECK-NEXT: %1 = llvm.call @ath_get_sub_tensor(%arg3, %0) : (!llvm.i64, !llvm<"i8*">) -> !llvm<"i8*">
// CHECK-NEXT: %2 = llvm.call @ath_get_sub_tensor(%arg3, %arg0) : (!llvm.i64, !llvm<"i8*">) -> !llvm<"i8*">
// CHECK-NEXT: %3 = llvm.call @ath_get_sub_tensor(%arg3, %arg1) : (!llvm.i64, !llvm<"i8*">) -> !llvm<"i8*">
// CHECK-NEXT: llvm.call @ath_allocate(%1) : (!llvm<"i8*">) -> ()
// CHECK-NEXT: llvm.call @ath_lock_tensor(%2) : (!llvm<"i8*">) -> ()
// CHECK-NEXT: llvm.call @ath_lock_tensor(%3) : (!llvm<"i8*">) -> ()
// CHECK-NEXT: llvm.call @ath_lock_tensor(%1) : (!llvm<"i8*">) -> ()
// CHECK-NEXT: %4 = llvm.mlir.constant(1.000000e+00 : f32) : !llvm.float
// CHECK-NEXT: %5 = llvm.mlir.constant(1 : i32) : !llvm.i32
// CHECK-NEXT: %6 = llvm.alloca %5 x !llvm<"i8*"> {alignment = 8 : i64} : (!llvm.i32) -> !llvm<"{ i8*, i32, { i64, i8*, i32 }*, i64, i64*, i64* }">
// CHECK-NEXT: %7 = llvm.mlir.constant(0 : i32) : !llvm.i32
// CHECK-NEXT: %8 = llvm.getelementptr %6[%7, %7] : (!llvm<"{ i8*, i32, { i64, i8*, i32 }*, i64, i64*, i64* }">, !llvm.i32, !llvm.i32) -> !llvm<"i8**">
// CHECK-NEXT: %9 = llvm.mlir.addressof @ath_graph.add : !llvm<"[13 x i8]*">
// CHECK-NEXT: %10 = llvm.getelementptr %9[%7, %7] : (!llvm<"[13 x i8]*">, !llvm.i32, !llvm.i32) -> !llvm<"i8*">
// CHECK-NEXT: llvm.store %10, %8 : !llvm<"i8**">
// CHECK-NEXT: %11 = llvm.getelementptr %6[%7, %5] : (!llvm<"{ i8*, i32, { i64, i8*, i32 }*, i64, i64*, i64* }">, !llvm.i32, !llvm.i32) -> !llvm<"i32*">
// CHECK-NEXT: %12 = llvm.mlir.constant(5 : i32) : !llvm.i32
// CHECK-NEXT: llvm.store %12, %11 : !llvm<"i32*">
// CHECK-NEXT: %13 = llvm.alloca %12 x !llvm.i64 {alignment = 16 : i64} : (!llvm.i32) -> !llvm<"{ i64, i8*, i32 }">
// CHECK-NEXT: %14 = llvm.mlir.constant(0 : i32) : !llvm.i32
// CHECK-NEXT: %15 = llvm.mlir.constant(1 : i32) : !llvm.i32
// CHECK-NEXT: %16 = llvm.mlir.constant(2 : i32) : !llvm.i32
// CHECK-NEXT: %17 = llvm.mlir.constant(0 : i32) : !llvm.i32
// CHECK-NEXT: %18 = llvm.getelementptr %13[%14, %17] : (!llvm<"{ i64, i8*, i32 }">, !llvm.i32, !llvm.i32) -> !llvm<"{ i64, i8*, i32 }">
// CHECK-NEXT: %19 = llvm.getelementptr %18[%14, %15] : (!llvm<"{ i64, i8*, i32 }">, !llvm.i32, !llvm.i32) -> !llvm<"i8**">
// CHECK-NEXT: %20 = llvm.getelementptr %18[%14, %16] : (!llvm<"{ i64, i8*, i32 }">, !llvm.i32, !llvm.i32) -> !llvm<"i32*">
// CHECK-NEXT: llvm.store %2, %19 : !llvm<"i8**">
// CHECK-NEXT: llvm.store %14, %20 : !llvm<"i32*">
// CHECK-NEXT: %21 = llvm.mlir.constant(1 : i32) : !llvm.i32
// CHECK-NEXT: %22 = llvm.getelementptr %13[%14, %21] : (!llvm<"{ i64, i8*, i32 }">, !llvm.i32, !llvm.i32) -> !llvm<"{ i64, i8*, i32 }">
// CHECK-NEXT: %23 = llvm.getelementptr %22[%14, %15] : (!llvm<"{ i64, i8*, i32 }">, !llvm.i32, !llvm.i32) -> !llvm<"i8**">
// CHECK-NEXT: %24 = llvm.getelementptr %22[%14, %16] : (!llvm<"{ i64, i8*, i32 }">, !llvm.i32, !llvm.i32) -> !llvm<"i32*">
// CHECK-NEXT: %25 = llvm.alloca %15 x !llvm.float {alignment = 32 : i64} : (!llvm.i32) -> !llvm<"float*">
// CHECK-NEXT: llvm.store %4, %25 : !llvm<"float*">
// CHECK-NEXT: %26 = llvm.bitcast %25 : !llvm<"float*"> to !llvm<"i8*">
// CHECK-NEXT: llvm.store %26, %23 : !llvm<"i8**">
// CHECK-NEXT: llvm.store %15, %24 : !llvm<"i32*">
// CHECK-NEXT: %27 = llvm.getelementptr %22[%14, %14] : (!llvm<"{ i64, i8*, i32 }">, !llvm.i32, !llvm.i32) -> !llvm<"i64*">
// CHECK-NEXT: %28 = llvm.mlir.constant(4 : i64) : !llvm.i64
// CHECK-NEXT: llvm.store %28, %27 : !llvm<"i64*">
// CHECK-NEXT: %29 = llvm.mlir.constant(2 : i32) : !llvm.i32
// CHECK-NEXT: %30 = llvm.getelementptr %13[%14, %29] : (!llvm<"{ i64, i8*, i32 }">, !llvm.i32, !llvm.i32) -> !llvm<"{ i64, i8*, i32 }">
// CHECK-NEXT: %31 = llvm.getelementptr %30[%14, %15] : (!llvm<"{ i64, i8*, i32 }">, !llvm.i32, !llvm.i32) -> !llvm<"i8**">
// CHECK-NEXT: %32 = llvm.getelementptr %30[%14, %16] : (!llvm<"{ i64, i8*, i32 }">, !llvm.i32, !llvm.i32) -> !llvm<"i32*">
// CHECK-NEXT: llvm.store %3, %31 : !llvm<"i8**">
// CHECK-NEXT: llvm.store %14, %32 : !llvm<"i32*">
// CHECK-NEXT: %33 = llvm.mlir.constant(3 : i32) : !llvm.i32
// CHECK-NEXT: %34 = llvm.getelementptr %13[%14, %33] : (!llvm<"{ i64, i8*, i32 }">, !llvm.i32, !llvm.i32) -> !llvm<"{ i64, i8*, i32 }">
// CHECK-NEXT: %35 = llvm.getelementptr %34[%14, %15] : (!llvm<"{ i64, i8*, i32 }">, !llvm.i32, !llvm.i32) -> !llvm<"i8**">
// CHECK-NEXT: %36 = llvm.getelementptr %34[%14, %16] : (!llvm<"{ i64, i8*, i32 }">, !llvm.i32, !llvm.i32) -> !llvm<"i32*">
// CHECK-NEXT: %37 = llvm.alloca %15 x !llvm.float {alignment = 32 : i64} : (!llvm.i32) -> !llvm<"float*">
// CHECK-NEXT: llvm.store %4, %37 : !llvm<"float*">
// CHECK-NEXT: %38 = llvm.bitcast %37 : !llvm<"float*"> to !llvm<"i8*">
// CHECK-NEXT: llvm.store %38, %35 : !llvm<"i8**">
// CHECK-NEXT: llvm.store %15, %36 : !llvm<"i32*">
// CHECK-NEXT: %39 = llvm.getelementptr %34[%14, %14] : (!llvm<"{ i64, i8*, i32 }">, !llvm.i32, !llvm.i32) -> !llvm<"i64*">
// CHECK-NEXT: %40 = llvm.mlir.constant(4 : i64) : !llvm.i64
// CHECK-NEXT: llvm.store %40, %39 : !llvm<"i64*">
// CHECK-NEXT: %41 = llvm.mlir.constant(4 : i32) : !llvm.i32
// CHECK-NEXT: %42 = llvm.getelementptr %13[%14, %41] : (!llvm<"{ i64, i8*, i32 }">, !llvm.i32, !llvm.i32) -> !llvm<"{ i64, i8*, i32 }">
// CHECK-NEXT: %43 = llvm.getelementptr %42[%14, %15] : (!llvm<"{ i64, i8*, i32 }">, !llvm.i32, !llvm.i32) -> !llvm<"i8**">
// CHECK-NEXT: %44 = llvm.getelementptr %42[%14, %16] : (!llvm<"{ i64, i8*, i32 }">, !llvm.i32, !llvm.i32) -> !llvm<"i32*">
// CHECK-NEXT: llvm.store %1, %43 : !llvm<"i8**">
// CHECK-NEXT: llvm.store %14, %44 : !llvm<"i32*">
// CHECK-NEXT: %45 = llvm.mlir.constant(2 : i32) : !llvm.i32
// CHECK-NEXT: %46 = llvm.getelementptr %13[%7, %7] : (!llvm<"{ i64, i8*, i32 }">, !llvm.i32, !llvm.i32) -> !llvm<"{ i64, i8*, i32 }*">
// CHECK-NEXT: %47 = llvm.getelementptr %6[%7, %45] : (!llvm<"{ i8*, i32, { i64, i8*, i32 }*, i64, i64*, i64* }">, !llvm.i32, !llvm.i32) -> !llvm<"{ i64, i8*, i32 }**">
// CHECK-NEXT: llvm.store %46, %47 : !llvm<"{ i64, i8*, i32 }**">
// CHECK-NEXT: %48 = llvm.mlir.constant(1 : i64) : !llvm.i64
// CHECK-NEXT: %49 = llvm.mlir.constant(3 : i32) : !llvm.i32
// CHECK-NEXT: %50 = llvm.getelementptr %6[%7, %49] : (!llvm<"{ i8*, i32, { i64, i8*, i32 }*, i64, i64*, i64* }">, !llvm.i32, !llvm.i32) -> !llvm<"i64*">
// CHECK-NEXT: llvm.store %48, %50 : !llvm<"i64*">
// CHECK-NEXT: %51 = llvm.alloca %48 x !llvm.i64 {alignment = 16 : i64} : (!llvm.i64) -> !llvm<"[1 x i64]">
// CHECK-NEXT: %52 = llvm.mlir.constant(0 : i32) : !llvm.i32
// CHECK-NEXT: %53 = llvm.getelementptr %6[%7, %52] : (!llvm<"{ i8*, i32, { i64, i8*, i32 }*, i64, i64*, i64* }">, !llvm.i32, !llvm.i32) -> !llvm<"i64*">
// CHECK-NEXT: %54 = llvm.mlir.constant(8 : i64) : !llvm.i64
// CHECK-NEXT: llvm.store %54, %53 : !llvm<"i64*">
// CHECK-NEXT: %55 = llvm.mlir.constant(4 : i32) : !llvm.i32
// CHECK-NEXT: %56 = llvm.getelementptr %51[%7, %7] : (!llvm<"[1 x i64]">, !llvm.i32, !llvm.i32) -> !llvm<"i64*">
// CHECK-NEXT: %57 = llvm.getelementptr %6[%7, %55] : (!llvm<"{ i8*, i32, { i64, i8*, i32 }*, i64, i64*, i64* }">, !llvm.i32, !llvm.i32) -> !llvm<"i64**">
// CHECK-NEXT: llvm.store %56, %57 : !llvm<"i64**">
// CHECK-NEXT: %58 = llvm.alloca %48 x !llvm.i64 {alignment = 16 : i64} : (!llvm.i64) -> !llvm<"[1 x i64]">
// CHECK-NEXT: %59 = llvm.mlir.constant(0 : i32) : !llvm.i32
// CHECK-NEXT: %60 = llvm.getelementptr %6[%7, %59] : (!llvm<"{ i8*, i32, { i64, i8*, i32 }*, i64, i64*, i64* }">, !llvm.i32, !llvm.i32) -> !llvm<"i64*">
// CHECK-NEXT: %61 = llvm.mlir.constant(0 : i64) : !llvm.i64
// CHECK-NEXT: llvm.store %61, %60 : !llvm<"i64*">
// CHECK-NEXT: %62 = llvm.mlir.constant(5 : i32) : !llvm.i32
// CHECK-NEXT: %63 = llvm.getelementptr %58[%7, %7] : (!llvm<"[1 x i64]">, !llvm.i32, !llvm.i32) -> !llvm<"i64*">
// CHECK-NEXT: %64 = llvm.getelementptr %6[%7, %62] : (!llvm<"{ i8*, i32, { i64, i8*, i32 }*, i64, i64*, i64* }">, !llvm.i32, !llvm.i32) -> !llvm<"i64**">
// CHECK-NEXT: llvm.store %63, %64 : !llvm<"i64**">
// CHECK-NEXT: llvm.call @ath_release_tensor(%1) : (!llvm<"i8*">) -> ()
// CHECK-NEXT: llvm.call @ath_release_tensor(%2) : (!llvm<"i8*">) -> ()
// CHECK-NEXT: llvm.call @ath_release_tensor(%3) : (!llvm<"i8*">) -> ()
// CHECK-NEXT: llvm.return %0 : !llvm<"i8*">
// CHECK-NEXT: }
// CHECK-NEXT: llvm.func @SampleGraph(%arg0: !llvm<"i8*">, %arg1: !llvm.i64, %arg2: !llvm<"i8*">) {
// CHECK-NEXT: %0 = llvm.mlir.constant(0 : i64) : !llvm.i64
// CHECK-NEXT: %1 = llvm.call @ath_get_device_for_node(%0, %arg0) : (!llvm.i64, !llvm<"i8*">) -> !llvm<"i8*">
// CHECK-NEXT: %2 = llvm.call @InputA(%arg0, %arg1, %arg1, %1) : (!llvm<"i8*">, !llvm.i64, !llvm.i64, !llvm<"i8*">) -> !llvm.i8
// CHECK-NEXT: %3 = llvm.mlir.constant(1 : i64) : !llvm.i64
// CHECK-NEXT: %4 = llvm.call @ath_get_device_for_node(%3, %arg0) : (!llvm.i64, !llvm<"i8*">) -> !llvm<"i8*">
// CHECK-NEXT: %5 = llvm.call @InputB(%arg0, %arg1, %arg1, %4) : (!llvm<"i8*">, !llvm.i64, !llvm.i64, !llvm<"i8*">) -> !llvm.i8
// CHECK-NEXT: %6 = llvm.mlir.constant(2 : i64) : !llvm.i64
// CHECK-NEXT: %7 = llvm.call @ath_get_device_for_node(%6, %arg0) : (!llvm.i64, !llvm<"i8*">) -> !llvm<"i8*">
// CHECK-NEXT: %8 = llvm.call @SumNode(%2, %5, %arg0, %arg1, %arg1, %7) : (!llvm.i8, !llvm.i8, !llvm<"i8*">, !llvm.i64, !llvm.i64, !llvm<"i8*">) -> !llvm.i8
// CHECK-NEXT: llvm.return
// CHECK-NEXT: }
// CHECK-NEXT: }
