//===----------------------------------------------------------------------===//
// Copyright (c) 2020 PolarAI. All rights reserved.
//
// Licensed under MIT license.
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS, WITHOUT
// WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the
// License for the specific language governing permissions and limitations under
// the License.
//===----------------------------------------------------------------------===//

#pragma once

#include <polar_loaders_export.h>
#include <polarai/loaders/internal/DummyLoaderInternal.hpp>

namespace polarai::loaders {
class POLAR_LOADERS_EXPORT DummyLoader {
public:
  using InternalType = internal::DummyLoaderInternal;
};
} // namespace polarai::loaders
