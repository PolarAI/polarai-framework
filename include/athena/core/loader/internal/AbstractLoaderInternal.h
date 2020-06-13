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

#ifndef ATHENA_ABSTRACTLOADERIMPL_H
#define ATHENA_ABSTRACTLOADERIMPL_H

#include <athena/core/Entity.h>
#include <athena/core/context/internal/ContextInternal.h>
#include <athena/core/core_export.h>
#include <athena/core/loader/internal/TensorAllocator.h>
#include <athena/core/tensor/Accessor.h>
#include <athena/utils/string/StringView.h>

namespace athena::core::internal {
/**
 * Loaders is a concept that helps Athena put user data into Graph
 */
class ATH_CORE_EXPORT AbstractLoaderInternal : public Entity {
public:
  AbstractLoaderInternal(utils::WeakPtr<ContextInternal> context,
                         utils::Index publicIndex,
                         utils::String name = utils::String(""));

  virtual void load(Accessor<float>&) = 0;
  //virtual void load(Accessor<double>&) = 0;

protected:
  utils::String mName;
};
} // namespace athena::core::internal

#endif // ATHENA_ABSTRACTLOADER_H
