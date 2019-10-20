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

#ifndef ATHENA_MATRIX_H
#define ATHENA_MATRIX_H

#include <athena/backend/llvm/device/Device.h>
#include <athena/core/Allocator.h>
#include <athena/core/inner/Tensor.h>

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
extern void hadamard(athena::backend::llvm::Device *,
                     athena::core::Allocator *allocator,
                     athena::core::inner::Tensor *a,
                     T scaleA,
                     athena::core::inner::Tensor *b,
                     T scaleB,
                     athena::core::inner::Tensor *c);

namespace athena::backend::llvm {
template <typename T>
struct GEMMOptions {
    bool transposeA;
    bool transposeB;
    T alpha;
    T beta;
};
}  // namespace athena::backend::llvm

template <typename T>
extern void gemm(athena::backend::llvm::Device *,
                 athena::core::Allocator *allocator,
                 athena::backend::llvm::GEMMOptions<T> *options,
                 athena::core::inner::Tensor *a,
                 athena::core::inner::Tensor *b,
                 athena::core::inner::Tensor *c);

#endif  // ATHENA_MATRIX_H
