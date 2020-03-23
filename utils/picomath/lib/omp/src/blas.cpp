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

#include <kernels/blas/gemm.h>

#include <picomath/blas/gemm.h>
#include <support/Index.hpp>
#include <support/kernel.hpp>

using namespace picomath;

extern "C" {
void cblas_sgemm(const CBLAS_ORDER order, const CBLAS_TRANSPOSE transposeA,
                 const CBLAS_TRANSPOSE transposeB, const int M, const int N,
                 const int K, const float alpha, const float* inpA,
                 const int lda, const float* inpB, const int ldb,
                 const float beta, float* outC, const int ldc) {
  global_accessor<float, 2, access_mode::read> bufA(
      const_cast<float*>(inpA),
      {static_cast<size_t>(lda), static_cast<size_t>(K)});
  global_accessor<float, 2, access_mode::read> bufB(
      const_cast<float*>(inpB),
      {static_cast<size_t>(ldb), static_cast<size_t>(N)});
  global_accessor<float, 2, access_mode::write> bufC(
      outC, {static_cast<size_t>(ldc), static_cast<size_t>(N)});

  GemmKernel kernel(bufA, transposeA == CblasTrans, bufB,
                           transposeB == CblasTrans, bufC, M, N, K);
  runKernel<decltype(kernel), 2>(
      kernel, {static_cast<size_t>(M), static_cast<size_t>(N)});
}
}
