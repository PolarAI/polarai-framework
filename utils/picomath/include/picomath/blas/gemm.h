//===----------------------------------------------------------------------===//
// Copyright (c) 2020 Athena. All rights reserved.
// https://getathena.ml
//
// Licensed under MIT license.
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS, WITHOUT
// WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the
// License for the specific language governing permissions and limitations under
// the License.
//===----------------------------------------------------------------------===//

#ifndef ATHENA_GEMM_H
#define ATHENA_GEMM_H

#ifdef __cplusplus
extern "C" {
#endif
enum CBLAS_ORDER { CblasRowMajor, CblasColMajor };
enum CBLAS_TRANSPOSE { CblasNoTrans, CblasTrans, CblasConjTrans, AtlasConj };

/// Generic float matrix-matrix multiplication.
///
/// outC = alpha * inpA * inpB + beta * outC
///
/// \param order is row-major or column major matrix data order.
/// \param transposeA is matrix A transposition mode.
/// \param transposeB is matrix B transposition mode.
/// \param M is number of rows in matrices A and C.
/// \param N is number of columns in matrices B and C.
/// \param K is number of columns in matrix A, and number of rows in matrix B.
/// \param alpha is scaling factor for the product of matrices A and B.
/// \param inpA is a pointer to matrix A data.
/// \param lda is the size of first dimension of matrix A.
/// \param inpB is a pointer to matrix B data.
/// \param ldb is the size of first dimension of matrix B.
/// \param beta is scaling factor for matrix C.
/// \param outC is a pointer to matrix C data.
/// \param ldc is the size of first dimension of matrix C.
void cblas_sgemm(const CBLAS_ORDER order, const CBLAS_TRANSPOSE transposeA,
                 const CBLAS_TRANSPOSE transposeB, const int M, const int N,
                 const int K, const float alpha, const float* inpA,
                 const int lda, const float* inpB, const int ldb,
                 const float beta, float* outC, const int ldc);
#ifdef __cplusplus
}
#endif

#endif // ATHENA_GEMM_H
