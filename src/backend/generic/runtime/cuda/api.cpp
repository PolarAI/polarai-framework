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

#include "CudaContext.hpp"

#include <polar_rt_cuda_export.h>
#include <polarai/backend/generic/runtime/api.h>

using namespace polarai::backend::generic;

extern "C" {

POLAR_RT_CUDA_EXPORT Context* initContext() { return new CudaContext(); }

POLAR_RT_CUDA_EXPORT void closeContext(Context* ctx) { delete ctx; }
}
