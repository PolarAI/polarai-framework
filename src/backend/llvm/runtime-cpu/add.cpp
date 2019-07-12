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

#include <athena/backend/llvm/runtime/add.h>
#include <athena/backend/llvm/device/Device.h>
#include <athena/core/inner/Tensor.h>

#include <iostream>

using namespace athena::backend::llvm;
using namespace athena::core::inner;

void athena_fadd(void *a, size_t ca, void *b, size_t cb, void *c) {
    auto *af = static_cast<float *>(a);
    auto *bf = static_cast<float *>(b);
    auto *cf = static_cast<float *>(c);

    for (size_t i = 0; i < ca; i++) {
        cf[i] = af[i] + bf[i];
    }
}

template <typename T>
void add(Device *, Tensor *a, Tensor *b) {

}