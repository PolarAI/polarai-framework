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

#include <picomath/blas/gemm.h>
#include <picomath/omp/Accessor.h>
#include <picomath/omp/Index.h>
#include <picomath/omp/kernel.h>

#include "kernels/blas/gemm.h"

template <typename AccessorAT, typename AccessorBT, typename AccessorCT>
GemmKernel(AccessorAT, bool, AccessorBT, bool, AccessorCT, size_t, size_t,
           size_t)
    ->GemmKernel<AccessorAT, AccessorBT, AccessorCT, Index<2>>;

using namespace picomath;

extern "C" {
void cblas_sgemm(const CBLAS_ORDER order, const CBLAS_TRANSPOSE transposeA,
                 const CBLAS_TRANSPOSE transposeB, const int M, const int N,
                 const int K, const float alpha, const float* inpA,
                 const int lda, const float* inpB, const int ldb,
                 const float beta, float* outC, const int ldc) {
  Accessor<const float, 2> bufA(
      inpA, {static_cast<size_t>(lda), static_cast<size_t>(K)});
  Accessor<const float, 2> bufB(
      inpB, {static_cast<size_t>(ldb), static_cast<size_t>(N)});
  Accessor<float, 2> bufC(outC,
                          {static_cast<size_t>(ldc), static_cast<size_t>(N)});

  Index<2> range{static_cast<size_t>(M), static_cast<size_t>(N)};

  GemmKernel kernel(bufA, transposeA == CblasTrans, bufB,
                    transposeB == CblasTrans, bufC, M, N, K);
  runKernel<decltype(kernel), 2>(
      kernel, {static_cast<size_t>(M), static_cast<size_t>(N)});
}
}
