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

#ifndef ATHENA_COPYLOADER_H
#define ATHENA_COPYLOADER_H

#include <athena/core/loader/AbstractLoader.h>
#include <athena/loaders/internal/CopyLoaderInternal.h>

namespace athena::loaders {
namespace internal {
class CopyLoaderInternal;
}
class ATH_LOADERS_EXPORT AbstractLoader {
public:
  using InternalType = internal::CopyLoaderInternal;
};
} // namespace athena::loaders

#endif // ATHENA_COPYLOADER_H
