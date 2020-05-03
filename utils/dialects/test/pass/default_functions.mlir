// RUN: athena-opt --deploy-default-functions %s | FileCheck %s

module {
// CHECK: llvm.func @ath_allocate(!llvm<"void*">, !llvm<"void*">, !llvm<"void*">)
// CHECK: llvm.func @ath_release_tensor(!llvm<"void*">, !llvm<"void*">, !llvm<"void*">)
// CHECK: llvm.func @ath_get_tensor_ptr(!llvm<"void*">, !llvm<"void*">, !llvm.i64) -> !llvm<"void*">
// CHECK: llvm.func @ath_lock_tensor(!llvm<"void*">, !llvm<"void*">, !llvm<"void*">, !llvm.i32)
// CHECK: llvm.func @ath_get_sub_tensor(!llvm.i64, !llvm<"void*">, !llvm.i32) -> !llvm<"void*">
}