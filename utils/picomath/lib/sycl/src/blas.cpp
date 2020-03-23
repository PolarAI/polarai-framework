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

#include "kernels/blas/gemm.h"
#include <mkl_blas_sycl.hpp>

using namespace cl::sycl;

template <typename T>
void gemm(cl::sycl::queue& queue, transpose transa, transpose transb, int64_t m,
          int64_t n, int64_t k, T alpha, cl::sycl::buffer<T, 1>& a, int64_t lda,
          cl::sycl::buffer<T, 1>& b, int64_t ldb, T beta,
          cl::sycl::buffer<T, 1>& c, int64_t ldc) {
  auto bufInpA = a.template reinterpret<T, 2>({lda, k});
  auto bufInpB = b.template reinterpret<T, 2>({ldb, n});
  auto bufOutC = c.template reinterpret<T, 2>({ldc, n});

  queue.submit([&](handler& cgh) {
    auto accInpA = bufInpA.get_access<cl::sycl::access::mode::read>(cgh);
    auto accInpB = bufInpB.get_access<cl::sycl::access::mode::read>(cgh);
    auto accOutC = bufOutC.get_access<cl::sycl::access::mode::write>(cgh);

    picomath::GemmKernel kernel(accInpA, transa == transpose::T, accInpB,
                                transb == transpose::T, accOutC, m, n, k);
    cgh.parallel_for(range<2>(m, n), kernel);
  });
}
