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

#include <polarai/core/Generator.hpp>
#include <polar_backend_generic_export.h>

namespace mlir {
class OpBuilder;
}

namespace polarai::backend::generic {
/// Feeds Generator with functors to generate correct MLIR.
POLAR_BACKEND_GENERIC_EXPORT void
populateCodeGenPatterns(core::internal::Generator& generator,
                        mlir::OpBuilder& builder);
} // namespace athena::backend::llvm
