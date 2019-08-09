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

#include <athena/backend/llvm/device/Device.h>
#include <athena/backend/llvm/runtime/matrix.h>
#include <athena/core/Allocator.h>
#include <athena/core/inner/Tensor.h>

using namespace athena::backend::llvm;
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
              Tensor *a,
              T scaleA,
              Tensor *b,
              T scaleB,
              Tensor *c) {
    auto *ap = reinterpret_cast<T *>(allocator->getRAMPointer(*a));
    auto *bp = reinterpret_cast<T *>(allocator->getRAMPointer(*b));
    auto *cp = reinterpret_cast<T *>(allocator->getRAMPointer(*c));

    for (size_t i = 0; i < c->getShapeView().getTotalSize(); i++) {
        cp[i] = scaleA * ap[i] * scaleB * bp[i];
    }
}

template void hadamard<float>(Device *,
                              Allocator *allocator,
                              Tensor *a,
                              float scaleA,
                              Tensor *b,
                              float scaleB,
                              Tensor *c);

template void hadamard<double>(Device *,
                               Allocator *allocator,
                               Tensor *a,
                               double scaleA,
                               Tensor *b,
                               double scaleB,
                               Tensor *c);