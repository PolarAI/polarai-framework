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

#ifndef ATHENA_DUMMYLOADERINTERNAL_H
#define ATHENA_DUMMYLOADERINTERNAL_H

#include <athena/core/loader/internal/AbstractLoaderInternal.h>
#include <athena/loaders/loaders_export.h>

namespace athena::loaders::internal {
class ATH_LOADERS_EXPORT DummyLoaderInternal
    : public core::internal::AbstractLoaderInternal {
public:
  DummyLoaderInternal(
      utils::WeakPtr<core::internal::ContextInternal> context,
      utils::Index publicIndex,
      utils::String name = utils::String(""));

  void load(core::Accessor<float>&) override;
  //void load(core::Accessor<double>&) override;
};
} // namespace athena::loaders::internal

#endif // ATHENA_DUMMYLOADERINTERNAL_H
