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

#include <athena/core/Generator.h>
#include <polar_backend_generic_export.h>

namespace mlir {
class OpBuilder;
}

namespace athena::backend::llvm {
/// Feeds Generator with functors to generate correct MLIR.
POLAR_BACKEND_GENERIC_EXPORT void
populateCodeGenPatterns(core::internal::Generator& generator,
                        mlir::OpBuilder& builder);
} // namespace athena::backend::llvm
