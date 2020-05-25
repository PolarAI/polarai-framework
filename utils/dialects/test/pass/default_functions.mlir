// RUN: athena-opt --deploy-default-functions %s | FileCheck %s

module {
  // CHECK: llvm.func @ath_allocate(!llvm<"i8*">, !llvm<"i8*">, !llvm<"i8*">)
  // CHECK: llvm.func @ath_release(!llvm<"i8*">, !llvm<"i8*">, !llvm<"i8*">)
  // CHECK: llvm.func @ath_lock(!llvm<"i8*">, !llvm<"i8*">, !llvm<"i8*">, !llvm.i32)
  // CHECK: llvm.func @ath_device_select(!llvm<"i8*">, !llvm.i64) -> !llvm<"i8*">
  // CHECK: llvm.func @ath_load(!llvm<"i8*">, !llvm.i64, !llvm<"i8*">)
  // CHECK: llvm.func @ath_barrier(!llvm.i64, !llvm<"i8*">)
  // CHECK: llvm.func @ath_launch(!llvm<"i8*">, !llvm<"i8*">, !llvm<"i8*">, !llvm<"{ i8*, i32, { i64, i8*, i32 }*, i64, i64*, i64* }*">) -> !llvm<"i8*">
}
