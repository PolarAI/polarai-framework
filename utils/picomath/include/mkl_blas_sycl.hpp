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

#ifndef ATHENA_MKL_BLAS_SYCL_HPP
#define ATHENA_MKL_BLAS_SYCL_HPP

#include <CL/sycl.hpp>

// Data types
enum class transpose : char {
  N = 0,
  nontrans = 0,
  T = 1,
  trans = 1,
  conjtrans = 3,
  C = 3
};

// BLAS Level 3

template <typename T>
void gemm(cl::sycl::queue& queue, transpose transa, transpose transb, int64_t m,
          int64_t n, int64_t k, T alpha, cl::sycl::buffer<T, 1>& a, int64_t lda,
          cl::sycl::buffer<T, 1>& b, int64_t ldb, T beta,
          cl::sycl::buffer<T, 1>& c, int64_t ldc);

#endif // ATHENA_MKL_BLAS_SYCL_HPP
