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

#ifndef ATHENA_MEMCPYLOADERINTERNAL_H
#define ATHENA_MEMCPYLOADERINTERNAL_H

#include <athena/core/loader/internal/AbstractLoaderInternal.h>
#include <athena/loaders/loaders_export.h>

namespace athena::loaders::internal {
class ATH_LOADERS_EXPORT MemcpyLoaderInternal
    : public core::internal::AbstractLoaderInternal {
public:
  MemcpyLoaderInternal(utils::WeakPtr<core::internal::ContextInternal> context,
                     utils::Index publicIndex, void* source, size_t len,
                     utils::String name = utils::String(""));

  void load(core::Accessor<float>&) override;
  //void load(core::Accessor<double>&) override;

  void setPointer(void* source, size_t size);

protected:
  void* mSource;
  size_t mLen;
};
} // namespace athena::loaders::internal

#endif // ATHENA_COPYLOADERINTERNAL_H
