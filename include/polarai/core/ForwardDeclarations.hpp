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

#include <polar_core_export.h>

#pragma once
namespace polarai::core {
class POLAR_CORE_EXPORT AbstractNode;
class POLAR_CORE_EXPORT Node;
class POLAR_CORE_EXPORT InputNode;
class POLAR_CORE_EXPORT OutputNodeInternal;
class POLAR_CORE_EXPORT LossNode;
namespace impl {
class POLAR_CORE_EXPORT GraphImpl;
class POLAR_CORE_EXPORT ContextImpl;

class POLAR_CORE_EXPORT AbstractNodeImpl;
class POLAR_CORE_EXPORT NodeImpl;
class POLAR_CORE_EXPORT OutputNodeImpl;
} // namespace impl
namespace internal {
class POLAR_CORE_EXPORT GraphInternal;
class POLAR_CORE_EXPORT ContextInternal;

class POLAR_CORE_EXPORT AbstractNodeInternal;
class POLAR_CORE_EXPORT NodeInternal;
class POLAR_CORE_EXPORT InputNodeInternal;
class POLAR_CORE_EXPORT OutputNodeInternal;
} // namespace internal
} // namespace polarai::core
