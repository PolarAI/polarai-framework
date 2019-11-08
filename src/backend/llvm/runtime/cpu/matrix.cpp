/*
 * Copyright (c) 2019 Athena. All rights reserved.
 * https://getathena.ml
 *
 * Licensed under MIT license.
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS, WITHOUT
 * WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the
 * License for the specific language governing permissions and limitations under
 * the License.
 */

#include <athena/backend/llvm/runtime/Device.h>
#include <athena/backend/llvm/runtime/structs.h>
#include <athena/core/Allocator.h>
#include <athena/core/inner/Tensor.h>

#ifdef ATHENA_APPLE_ACCELERATE
#include <Accelerate/Accelerate.h>
#else
#include <cblas.h>
#endif

using namespace athena::backend::llvm;
using namespace athena::backend;
using namespace athena::core::inner;
using namespace athena::core;

/**
 * Hadamard (element-wise) product
 * @tparam T Tensor content type
 * @param allocator Allocator
 * @param a First Tensor
 * @param scaleA First Tensor multiplier
 * @param b Second Tensor
 * @param scaleB Second Tensor multiplier
 * @param c Result Tensor
 */
template <typename T>
void hadamard(Device *,
              Allocator *allocator,
              HadamardOptions<T> *options,
              Tensor *a,
              Tensor *b,
              Tensor *c) {
    auto *ap = reinterpret_cast<T *>(allocator->getRAMPointer(*a));
    auto *bp = reinterpret_cast<T *>(allocator->getRAMPointer(*b));
    auto *cp = reinterpret_cast<T *>(allocator->getRAMPointer(*c));

    for (size_t i = 0; i < c->getShapeView().getTotalSize(); i++) {
        cp[i] = options->alpha * ap[i] * bp[i] + options->beta * cp[i];
    }
}

template void hadamard<float>(Device *,
                              Allocator *allocator,
                              HadamardOptions<float> *options,
                              Tensor *a,
                              Tensor *b,
                              Tensor *c);

template void hadamard<double>(Device *,
                               Allocator *allocator,
                               HadamardOptions<double> *options,
                               Tensor *a,
                               Tensor *b,
                               Tensor *c);

template <typename T>
void gemm(Device *,
          Allocator *allocator,
          GEMMOptions<T> *options,
          Tensor *a,
          Tensor *b,
          Tensor *c) {
    new FatalError(ATH_NOT_IMPLEMENTED, "Not implemented");
};

template <>
void gemm<float>(Device *,
                 Allocator *allocator,
                 GEMMOptions<float> *options,
                 Tensor *a,
                 Tensor *b,
                 Tensor *c) {
    CBLAS_TRANSPOSE transposeA =
        options->transposeA ? CblasTrans : CblasNoTrans;
    CBLAS_TRANSPOSE transposeB =
        options->transposeB ? CblasTrans : CblasNoTrans;

    auto aData = reinterpret_cast<float *>(allocator->getRAMPointer(*a));
    auto bData = reinterpret_cast<float *>(allocator->getRAMPointer(*b));
    auto cData = reinterpret_cast<float *>(allocator->getRAMPointer(*c));

    const int M =
        static_cast<const int>(a->getShape().dim(options->transposeA ? 0 : 1));
    const int K =
        static_cast<const int>(a->getShape().dim(options->transposeA ? 1 : 0));
    const int N =
        static_cast<const int>(b->getShape().dim(options->transposeB ? 1 : 0));

    cblas_sgemm(CBLAS_ORDER::CblasRowMajor, transposeA, transposeB, M, N, K,
                options->alpha, aData, options->transposeA ? M : K, bData,
                options->transposeB ? K : N, options->beta, cData, N);
}

template <>
void gemm<double>(Device *,
                  Allocator *allocator,
                  GEMMOptions<double> *options,
                  Tensor *a,
                  Tensor *b,
                  Tensor *c) {
    CBLAS_TRANSPOSE transposeA =
        options->transposeA ? CblasTrans : CblasNoTrans;
    CBLAS_TRANSPOSE transposeB =
        options->transposeB ? CblasTrans : CblasNoTrans;

    auto aData = reinterpret_cast<double *>(allocator->getRAMPointer(*a));
    auto bData = reinterpret_cast<double *>(allocator->getRAMPointer(*b));
    auto cData = reinterpret_cast<double *>(allocator->getRAMPointer(*c));

    const int M =
        static_cast<const int>(a->getShape().dim(options->transposeA ? 0 : 1));
    const int K =
        static_cast<const int>(a->getShape().dim(options->transposeA ? 1 : 0));
    const int N =
        static_cast<const int>(b->getShape().dim(options->transposeB ? 1 : 0));

    cblas_dgemm(CBLAS_ORDER::CblasRowMajor, transposeA, transposeB, M, N, K,
                options->alpha, aData, options->transposeA ? M : K, bData,
                options->transposeB ? K : N, options->beta, cData, N);
}