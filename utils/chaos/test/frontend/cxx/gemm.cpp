// clang-format off
// fixme: re-enable tests when CI env is updated
// RU N: %chaoscc -frontend-only -use-mlir -dump-mlir %t | %check %t
// this is a sanity check to ensure MLIR is correct
// RU N: %chaoscc -frontend-only -use-mlir -dump-mlir %t | %mlir_opt --canonicalize --

// CHECK: module {
// CHECK-NEXT: func @_Z5sgemmPKfS0_Pfiii(%arg0: memref<?xf32>, %arg1:
// memref<?xf32>, %arg2: memref<?xf32>, %arg3: i32, %arg4: i32, %arg5: i32) {
void sgemm(const float* a, const float* b, float* c, const int M, const int N,
           const int K) {
  // CHECK-NEXT: %0 = "std.constant"() {value = 0 : i32} : () -> i32
  // CHECK-NEXT: %1 = "std.index_cast"(%0) : (i32) -> index
  // CHECK-NEXT: %2 = "std.index_cast"(%arg3) : (i32) -> index
  // CHECK-NEXT: %3 = "std.constant"() {value = 1 : index} : () -> index
  // CHECK-NEXT: "loop.for"(%1, %2, %3) ( {
  // CHECK-NEXT: ^bb0(%arg6: index):	// no predecessors
  for (int i = 0; i < M; i++) {
    // CHECK-NEXT: %4 = "std.constant"() {value = 0 : i32} : () -> i32
    // CHECK-NEXT: %5 = "std.index_cast"(%4) : (i32) -> index
    // CHECK-NEXT: %6 = "std.index_cast"(%arg5) : (i32) -> index
    // CHECK-NEXT: %7 = "std.constant"() {value = 1 : index} : () -> index
    // CHECK-NEXT: "loop.for"(%5, %6, %7) ( {
    // CHECK-NEXT: ^bb0(%arg7: index):	// no predecessors
    for (int k = 0; k < K; k++) {
      // CHECK-NEXT: %8 = "std.constant"() {value = 0 : i32} : () -> i32
      // CHECK-NEXT: %9 = "std.index_cast"(%8) : (i32) -> index
      // CHECK-NEXT: %10 = "std.index_cast"(%arg4) : (i32) -> index
      // CHECK-NEXT: %11 = "std.constant"() {value = 1 : index} : () -> index
      // CHECK-NEXT: "loop.for"(%9, %10, %11) ( {
      // CHECK-NEXT: ^bb0(%arg8: index):	// no predecessors
      for (int j = 0; j < N; j++) {
        // CHECK-NEXT: %12 = "std.index_cast"(%arg6) : (index) -> i32
        // CHECK-NEXT: %13 = "std.muli"(%12, %arg3) : (i32, i32) -> i32
        // CHECK-NEXT: %14 = "std.index_cast"(%arg8) : (index) -> i32
        // CHECK-NEXT: %15 = "std.addi"(%13, %14) : (i32, i32) -> i32
        // CHECK-NEXT: %16 = "std.index_cast"(%15) : (i32) -> index
        // CHECK-NEXT: %17 = "std.load"(%arg2, %16) : (memref<?xf32>, index) -> f32 CHECK-NEXT: %18 = "std.constant"() {value = 1.000000e+00 : f32} : () -> f32 CHECK-NEXT: %19 = "std.index_cast"(%arg6) : (index) -> i32
        // CHECK-NEXT: %20 = "std.muli"(%19, %arg3) : (i32, i32) -> i32
        // CHECK-NEXT: %21 = "std.index_cast"(%arg7) : (index) -> i32
        // CHECK-NEXT: %22 = "std.addi"(%20, %21) : (i32, i32) -> i32
        // CHECK-NEXT: %23 = "std.index_cast"(%22) : (i32) -> index
        // CHECK-NEXT: %24 = "std.load"(%arg0, %23) : (memref<?xf32>, index) -> f32 CHECK-NEXT: %25 = "std.mulf"(%18, %24) : (f32, f32) -> f32
        // CHECK-NEXT: %26 = "std.index_cast"(%arg7) : (index) -> i32
        // CHECK-NEXT: %27 = "std.muli"(%26, %arg5) : (i32, i32) -> i32
        // CHECK-NEXT: %28 = "std.index_cast"(%arg8) : (index) -> i32
        // CHECK-NEXT: %29 = "std.addi"(%27, %28) : (i32, i32) -> i32
        // CHECK-NEXT: %30 = "std.index_cast"(%29) : (i32) -> index
        // CHECK-NEXT: %31 = "std.load"(%arg1, %30) : (memref<?xf32>, index) -> f32 CHECK-NEXT: %32 = "std.mulf"(%25, %31) : (f32, f32) -> f32
        // CHECK-NEXT: %33 = "std.addf"(%17, %32) : (f32, f32) -> f32
        // CHECK-NEXT: %34 = "std.index_cast"(%arg6) : (index) -> i32
        // CHECK-NEXT: %35 = "std.muli"(%34, %arg3) : (i32, i32) -> i32
        // CHECK-NEXT: %36 = "std.index_cast"(%arg8) : (index) -> i32
        // CHECK-NEXT: %37 = "std.addi"(%35, %36) : (i32, i32) -> i32
        // CHECK-NEXT: %38 = "std.index_cast"(%37) : (i32) -> index
        // CHECK-NEXT: "std.store"(%33, %arg2, %38) : (f32, memref<?xf32>, index) -> ()
        c[i * M + j] += 1.f * a[i * M + k] * b[k * K + j];
        // CHECK-NEXT: "loop.terminator"() : () -> ()
        // CHECK-NEXT: }) : (index, index, index) -> ()
      }
      // CHECK-NEXT: "loop.terminator"() : () -> ()
      // CHECK-NEXT: }) : (index, index, index) -> ()
    }
    // CHECK-NEXT: "loop.terminator"() : () -> ()
    // CHECK-NEXT: }) : (index, index, index) -> ()
  }
  // CHECK-NEXT: "std.return"() : () -> ()
  // CHECK-NEXT: }
}
// CHECK-NEXT: func @main() -> i32 {
int main() {
  // CHECK-NEXT: %0 = "std.constant"() {value = 2048 : i32} : () -> i32
  // CHECK-NEXT: %1 = "std.constant"() {value = 2048 : i32} : () -> i32
  // CHECK-NEXT: %2 = "std.muli"(%0, %1) : (i32, i32) -> i32
  // CHECK-NEXT: %3 = "std.index_cast"(%2) : (i32) -> index
  // CHECK-NEXT: %4 = "std.alloc"(%3) {alignment = 4 : i64} : (index) -> memref<?xf32>
  float* a = new float[2048 * 2048];
  // CHECK-NEXT: %5 = "std.constant"() {value = 2048 : i32} : () -> i32
  // CHECK-NEXT: %6 = "std.constant"() {value = 2048 : i32} : () -> i32
  // CHECK-NEXT: %7 = "std.muli"(%5, %6) : (i32, i32) -> i32
  // CHECK-NEXT: %8 = "std.index_cast"(%7) : (i32) -> index
  // CHECK-NEXT: %9 = "std.alloc"(%8) {alignment = 4 : i64} : (index) -> memref<?xf32>
  float* b = new float[2048 * 2048];
  // CHECK-NEXT: %10 = "std.constant"() {value = 2048 : i32} : () -> i32
  // CHECK-NEXT: %11 = "std.constant"() {value = 2048 : i32} : () -> i32
  // CHECK-NEXT: %12 = "std.muli"(%10, %11) : (i32, i32) -> i32
  // CHECK-NEXT: %13 = "std.index_cast"(%12) : (i32) -> index
  // CHECK-NEXT: %14 = "std.alloc"(%13) {alignment = 4 : i64} : (index) -> memref<?xf32>
  float* c = new float[2048 * 2048];
  // CHECK-NEXT: %15 = "std.constant"() {value = 2048 : i32} : () -> i32
  // CHECK-NEXT: %16 = "std.constant"() {value = 2048 : i32} : () -> i32
  // CHECK-NEXT: %17 = "std.constant"() {value = 2048 : i32} : () -> i32
  // CHECK-NEXT: "std.call"(%4, %9, %14, %15, %16, %17) {callee = @_Z5sgemmPKfS0_Pfiii} : (memref<?xf32>, memref<?xf32>, memref<?xf32>, i32, i32, i32) -> ()
  sgemm(a, b, c, 2048, 2048, 2048);
  // CHECK-NEXT: "std.dealloc"(%4) : (memref<?xf32>) -> ()
  delete[] a;
  // CHECK-NEXT: "std.dealloc"(%9) : (memref<?xf32>) -> ()
  delete[] b;
  // CHECK-NEXT: "std.dealloc"(%14) : (memref<?xf32>) -> ()
  delete[] c;
  // CHECK-NEXT: %18 = "std.constant"() {value = 0 : i32} : () -> i32
  // CHECK-NEXT: "std.return"(%18) : (i32) -> ()
  return 0;
  // CHECK-NEXT: }
}
// CHECK-NEXT: func @_Znwm(i64) -> memref<?xi64>
// CHECK-NEXT: func @_Znam(i64) -> memref<?xi64>
// CHECK-NEXT: func @_ZdlPv(memref<?xi64>)
// CHECK-NEXT: func @_ZdaPv(memref<?xi64>)
// CHECK-NEXT: }