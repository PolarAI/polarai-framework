// RUN: echo "1 2 3" | %check %t
// CHECK: 1 2 3
void sgemm(const float* a, const float* b, float* c, const int M, const int N,
           const int K) {
  for (int i = 0; i < M; i++) {
    for (int k = 0; k < K; k++) {
      for (int j = 0; j < N; j++)
        c[i * M + j] += 1.f * a[i * M + k] * b[k * K + j];
    }
  }
}
int main() {
  float* a = new float[2048 * 2048];
  float* b = new float[2048 * 2048];
  float* c = new float[2048 * 2048];
  sgemm(a, b, c, 2048, 2048, 2048);
  delete[] a;
  delete[] b;
  delete[] c;
  return 0;
}