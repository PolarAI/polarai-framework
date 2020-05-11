// RUN: athena-opt --deploy-default-functions %s | FileCheck %s

module {
// CHECK: llvm.func @ath_allocate(!llvm<"i8*">, !llvm<"i8*">, !llvm<"i8*">)
// CHECK: llvm.func @ath_release_tensor(!llvm<"i8*">, !llvm<"i8*">, !llvm<"i8*">)
// CHECK: llvm.func @ath_get_tensor_ptr(!llvm<"i8*">, !llvm<"i8*">, !llvm.i64) -> !llvm<"i8*">
// CHECK: llvm.func @ath_lock_tensor(!llvm<"i8*">, !llvm<"i8*">, !llvm<"i8*">, !llvm.i32)
// CHECK: llvm.func @ath_get_sub_tensor(!llvm.i64, !llvm<"i8*">, !llvm.i32) -> !llvm<"i8*">
// CHECK: llvm.func @ath_get_device_for_node(!llvm.i64, !llvm<"i8*">, !llvm.i32) -> !llvm<"i8*">
// CHECK: llvm.func @ath_launch(!llvm<"i8*">, !llvm<"i8*">, !llvm<"{ i8*, i32, { i64, i8*, i32 }*, i64, i64*, i64* }">)
}