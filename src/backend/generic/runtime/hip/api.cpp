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

#include "HIPContext.h"

#include <athena/backend/llvm/runtime/api.h>
#include <polar_rt_hip_export.h>

using namespace athena::backend::llvm;

extern "C" {

POLAR_RT_HIP_EXPORT Context* initContext() { return new HIPContext(); }

POLAR_RT_HIP_EXPORT void closeContext(Context* ctx) { delete ctx; }
}
