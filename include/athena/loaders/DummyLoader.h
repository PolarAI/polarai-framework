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

#ifndef ATHENA_DUMMYLOADER_H
#define ATHENA_DUMMYLOADER_H

#include <athena/loaders/internal/DummyLoaderInternal.h>

namespace athena::loaders {
class ATH_LOADERS_EXPORT DummyLoader {
public:
using InternalType = internal::DummyLoaderInternal;
};
} // namespace athena::operation

#endif // ATHENA_COPYOPERATION_H
