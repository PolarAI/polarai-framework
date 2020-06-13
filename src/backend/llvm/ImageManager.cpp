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

// #include "images.hpp"
#include "runtime/opencl/kernels.h"
#include "ImageManager.h"

#include <athena/backend/llvm/runtime/ProgramDesc.h>

namespace athena::backend::llvm {
auto getOpenCLSPIRVProgram() -> ProgramDesc {
  ProgramDesc prog;
#ifdef HAS_SPIRV
  prog.type = ProgramDesc::SPIRV;
  prog.length = kernels_spirv.size();
  prog.data = reinterpret_cast<const char*>(kernels_spirv.data());
#endif
  return prog;
}
auto getOpenCLTextProgram() -> ProgramDesc {
  ProgramDesc prog;
  prog.type = ProgramDesc::TEXT;
  prog.length = textKernels.size();
  prog.data = textKernels.data();
  return prog;
}
}
