// clang-format off
// RU N: %chaoscc -frontend-only -use-mlir -dump-mlir %t | %check %t
// RU N: %chaoscc -frontend-only -use-mlir -dump-mlir %t | %mlir_opt %mlir_opt --canonicalize --

// CHECK: module {
// CHECK: "clang.struct_decl"() {structName = "_Z4Test", structType =
// !clang.struct<i32>} : () -> ()
struct Test {
  int a;
};

// CHECK: func @_ZN4TestC1Ev(%arg0: !clang.ptr<!clang.struct<i32>>) {
// CHECK: "std.return"() : () -> ()
// CHECK: }
// CHECK: func @_ZN4TestC1ERKS_(!clang.ptr<!clang.struct<i32>>,
// !clang.ptr<!clang.struct<i32>>) CHECK: func
// @_ZN4TestC1EOS_(!clang.ptr<!clang.struct<i32>>,
// !clang.ptr<!clang.struct<i32>>)

// CHEKC: func @_Z3foov() {
void foo() {
  // CHECK: %0 = "std.constant"() {value = 1 : index} : () -> index
  // CHECK: %1 = "clang.alloca"(%0) : (index) -> !clang.ptr<!clang.struct<i32>>
  Test t;
  // CHECK: %2 = "clang.getelementprt"(%1) {idx = 0 : i64} :
  // (!clang.ptr<!clang.struct<i32>>) -> !clang.ptr<!clang.struct<i32>> CHECK:
  // "clang.store"(%2, %3) : (!clang.ptr<!clang.struct<i32>>, i32) -> ()
  t.a = 5;
  // CHECK: "std.return"() : () -> ()
}
