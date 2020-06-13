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

#ifndef CLANG_EXTRAS_COMPUTEDIALECT_H
#define CLANG_EXTRAS_COMPUTEDIALECT_H

#include "mlir/IR/Dialect.h"

namespace mlir::compute {

#include "Compute/ComputeOpsDialect.h.inc"
}

#endif // CLANG_EXTRAS_COMPUTEDIALECT_H
