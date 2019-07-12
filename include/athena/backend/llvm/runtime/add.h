/*
 * Copyright (c) 2019 Athena. All rights reserved.
 * https://getathena.ml
 *
 * Licensed under MIT license.
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an “AS IS” BASIS, WITHOUT
 * WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the
 * License for the specific language governing permissions and limitations under
 * the License.
 */
#ifndef ATHENA_ADD_H
#define ATHENA_ADD_H

#include <athena/core/inner/Tensor.h>
#include <athena/backend/llvm/device/Device.h>


#if __cplusplus
#include <cstddef>


extern "C" {
#else
#include <stddef.h>
#endif

void athena_fadd(void *a, size_t ca, void *b, size_t cb, void *c);

#if __cplusplus
}
#endif

template<typename T>
void add(
    athena::backend::llvm::Device *,
    athena::core::inner::Tensor *a,
    athena::core::inner::Tensor *b
);

template <>
void add<float>(
    athena::backend::llvm::Device *,
    athena::core::inner::Tensor *a,
    athena::core::inner::Tensor *b
);

template <>
void add<double>(
    athena::backend::llvm::Device *,
    athena::core::inner::Tensor *a,
    athena::core::inner::Tensor *b
);

#endif  // ATHENA_ADD_H
