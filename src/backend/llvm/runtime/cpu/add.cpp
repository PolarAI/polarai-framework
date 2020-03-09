/*
 * Copyright (c) 2018 Athena. All rights reserved.
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

#include <athena/backend/llvm/runtime/Device.h>
#include <athena/backend/llvm/runtime/add.h>
#include <athena/core/inner/Tensor.h>

using namespace athena::backend::llvm;
using namespace athena::core::inner;
using namespace athena::core;

template <typename T>
void add(Device* device, BackendAllocator* allocator, Tensor* a, Tensor* b,
         Tensor* c) {
  auto* af = allocator->get<T>(*a, *device);
  auto* bf = allocator->get<T>(*b, *device);
  auto* cf = allocator->get<T>(*c, *device);

  for (size_t i = 0; i < c->getShape().getTotalSize(); i++) {
    cf[i] =
        af[i]
        + bf[i];
  }
}

template <typename T>
void fma(athena::backend::llvm::Device* device, BackendAllocator* allocator,
         athena::core::inner::Tensor* a, T scaleA,
         athena::core::inner::Tensor* b, T scaleB,
         athena::core::inner::Tensor* c) {
  auto* af = reinterpret_cast<T*>(allocator->get<void*>(*a, *device));
  auto* bf = reinterpret_cast<T*>(allocator->get<void*>(*b, *device));
  auto* cf = reinterpret_cast<T*>(allocator->get<void*>(*c, *device));

  for (size_t i = 0; i < c->getShape().getTotalSize(); i++) {
    cf[i] = scaleA * af[i] + scaleB * bf[i];
  }
}

template void add<float>(athena::backend::llvm::Device*,
                         BackendAllocator*,
                         athena::core::inner::Tensor* a,
                         athena::core::inner::Tensor* b,
                         athena::core::inner::Tensor* c);

template void add<double>(athena::backend::llvm::Device*,
                          BackendAllocator*,
                          athena::core::inner::Tensor* a,
                          athena::core::inner::Tensor* b,
                          athena::core::inner::Tensor* c);

template void fma<float>(athena::backend::llvm::Device*,
                         BackendAllocator*,
                         athena::core::inner::Tensor* a, float scaleA,
                         athena::core::inner::Tensor* b, float scaleB,
                         athena::core::inner::Tensor* c);

template void fma<double>(athena::backend::llvm::Device*,
                          BackendAllocator*,
                          athena::core::inner::Tensor* a, double scaleA,
                          athena::core::inner::Tensor* b, double scaleB,
                          athena::core::inner::Tensor* c);
