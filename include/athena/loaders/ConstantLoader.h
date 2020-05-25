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

#ifndef ATHENA_CONSTANTLOADER_H
#define ATHENA_CONSTANTLOADER_H

#include <athena/core/loader/AbstractLoader.h>
#include <athena/loaders/internal/ConstantLoaderInternal.h>

namespace athena::loaders {
namespace internal {
class ConstantLoaderInternal;
}
class ATH_LOADERS_EXPORT AbstractLoader {
public:
  using InternalType = internal::ConstantLoaderInternal;
};
} // namespace athena::loaders

#endif // ATHENA_CONSTANTLOADER_H
