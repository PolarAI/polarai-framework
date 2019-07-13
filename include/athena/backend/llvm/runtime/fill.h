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

#ifndef ATHENA_FILL_H
#define ATHENA_FILL_H

#include <athena/backend/llvm/device/Device.h>
#include <athena/core/Allocator.h>
#include <athena/core/inner/Tensor.h>

template <typename T>
extern void fill(athena::backend::llvm::Device *,
                 athena::core::Allocator *,
                 athena::core::inner::Tensor *a,
                 T value);
#endif  // ATHENA_FILL_H
