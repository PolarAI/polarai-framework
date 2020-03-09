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
#include <athena/backend/llvm/BackendAllocator.h>
#include <athena/core/inner/Tensor.h>

using namespace athena::backend::llvm;
using namespace athena::core::inner;
using namespace athena::core;

template <typename T>
void mse(Device* device, BackendAllocator* allocator, Tensor* a, Tensor* b, Tensor* c) {
  auto* af = allocator->get<T>(*a, *device);
  auto* bf = allocator->get<T>(*b, *device);
  auto* cf = allocator->get<T>(*c, *device);

  for (size_t i = 0; i < a->getShape().getTotalSize(); i++) {
    T part = af[i] + bf[i];
    *cf += part * part;
  }
  *cf /= a->getShape().getTotalSize();
}

template void mse<float>(athena::backend::llvm::Device*,
                         BackendAllocator*,
                         athena::core::inner::Tensor* a,
                         athena::core::inner::Tensor* b,
                         athena::core::inner::Tensor* c);

template void mse<double>(athena::backend::llvm::Device*,
                          BackendAllocator*,
                          athena::core::inner::Tensor* a,
                          athena::core::inner::Tensor* b,
                          athena::core::inner::Tensor* c);