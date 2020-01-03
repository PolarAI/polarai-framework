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

#include <athena/backend/llvm/runtime/Device.h>
#include <athena/backend/llvm/runtime/structs.h>
#include <athena/core/Allocator.h>
#include <athena/core/inner/Tensor.h>

/**
 * Hadamard (element-wise) product
 *
 * c = options.alpha * a * b + options.beta * c
 *
 * @tparam T Tensor content type
 * @param allocator Allocator
 * @param options Multiplication options
 * @param a First Tensor
 * @param b Second Tensor
 * @param c Result Tensor
 */
template <typename T>
extern void
hadamard(athena::backend::llvm::Device*, athena::core::Allocator* allocator,
         athena::backend::HadamardOptions<T>* options,
         athena::core::inner::Tensor* a, athena::core::inner::Tensor* b,
         athena::core::inner::Tensor* c);

/**
 * General matrix-matrix multiplication
 *
 * c = options.alpha * a.b + options.beta * c
 *
 * @tparam T Tensor content type
 * @param allocator Allocator
 * @param options Multiplication options
 * @param a First Tensor
 * @param b Second Tensor
 * @param c Result Tensor
 */
template <typename T>
extern void
gemm(athena::backend::llvm::Device*, athena::core::Allocator* allocator,
     athena::backend::GEMMOptions<T>* options, athena::core::inner::Tensor* a,
     athena::core::inner::Tensor* b, athena::core::inner::Tensor* c);

#endif // ATHENA_MATRIX_H
