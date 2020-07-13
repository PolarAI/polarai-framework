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

#include "MetalContext.hh"

#include <polarai/backend/generic/runtime/api.hpp>
#include <polar_rt_metal_export.h>

using namespace polarai::backend::generic;

extern "C" {
POLAR_RT_METAL_EXPORT Context *initContext() { return new MetalContext(); }

POLAR_RT_METAL_EXPORT void closeContext(Context *ctx) { delete ctx; }
}
