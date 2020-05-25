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

#ifndef ATHENA_CONSTANTLOADERINTERNAL_H
#define ATHENA_CONSTANTLOADERINTERNAL_H

#include <athena/core/loader/internal/AbstractLoaderInternal.h>
#include <athena/loaders/loaders_export.h>

namespace athena::loaders::internal {
class ATH_LOADERS_EXPORT ConstantLoaderInternal
    : public core::internal::AbstractLoaderInternal {
public:
  ConstantLoaderInternal(
      utils::WeakPtr<core::internal::ContextInternal> context,
      utils::Index publicIndex, double value,
      utils::String name = utils::String(""));

protected:
  double mValue;
};
} // namespace athena::loaders::internal

#endif // ATHENA_CONSTANTLOADERINTERNAL_H
