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

#ifndef ATHENA_ARGINFO_H
#define ATHENA_ARGINFO_H

#include "mlir/IR/Operation.h"
#include "mlir/Transforms/DialectConversion.h"

/// Returns count of arguments that actual kernel takes.
///
/// Usually that's the same as MLIR Op::getNumOperands(), however, there may
/// be exceptions.
///
/// \param op is an MLIR AthenaGraphDialect operation
/// \return count of arguments.
uint32_t getArgsCount(mlir::Operation* op);

/// Fills the ArgDesc structure.
///
/// The information for the structure is mostly taken from operation arguments.
/// However, there may be exceptions.
///
/// \param argDescArray is a result of alloca command for ArgDesc array.
/// \param op is a builtin operation to take arguments from.
/// \param operands is a view into array of converted builtin operands.
/// \param rewriter is a ConversionPatternRewriter for current op.
void fillArgDesc(mlir::Value argDescArray, mlir::Operation* op,
                 llvm::ArrayRef<mlir::Value> operands,
                 mlir::ConversionPatternRewriter& rewriter);

#endif // ATHENA_ARGINFO_H
