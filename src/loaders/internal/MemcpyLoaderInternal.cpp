//===----------------------------------------------------------------------===//
// Copyright (c) 2020 Athena. All rights reserved.
// https://getathena.ml
//
// Licensed under MIT license.
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS, WITHOUT
// WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the
// License for the specific language governing permissions and limitations under
// the License.
//===----------------------------------------------------------------------===//

#include <athena/loaders/internal/MemcpyLoaderInternal.h>

#include <cstring>

namespace athena::loaders::internal {
MemcpyLoaderInternal::MemcpyLoaderInternal(
    utils::WeakPtr<core::internal::ContextInternal> context,
    utils::Index publicIndex, void* source, size_t len, utils::String name)
    : core::internal::AbstractLoaderInternal(std::move(context), publicIndex,
                                             std::move(name)),
      mSource(source), mLen(len) {}
void MemcpyLoaderInternal::load(core::Accessor<float>& acc){
  //std::cout << "MemcpyLoader: Loading to " << acc.getRawPtr() << "; Size: " << mLen << "; Source: " << mSource << std::endl;
  std::memcpy(acc.getRawPtr(), mSource, mLen);
}
//void MemcpyLoaderInternal::load(core::Accessor<double>& acc){
//  std::memcpy(acc.getRawPtr(), mSource, mLen);
//}
void MemcpyLoaderInternal::setPointer(void* source, size_t size) {
  mSource = source;
  mLen = size;
}
} // namespace athena::loaders::internal
