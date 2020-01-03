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

#include <athena/backend/llvm/runtime/allocate.h>
#include <athena/core/Allocator.h>
#include <athena/core/inner/Tensor.h>

using namespace athena::backend::llvm;
using namespace athena::core::inner;
using namespace athena::core;

template <typename T> void allocate(Device*, Allocator* allocator, Tensor* a) {
  allocator->allocate(*a);
}

template void allocate<void>(athena::backend::llvm::Device*,
                             athena::core::Allocator*,
                             athena::core::inner::Tensor* a);