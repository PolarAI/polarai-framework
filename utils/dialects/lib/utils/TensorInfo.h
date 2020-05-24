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

#pragma once

#include "mlir/Dialect/LLVMIR/LLVMDialect.h"

inline mlir::LLVM::LLVMType
getTensorInfoType(mlir::LLVM::LLVMDialect* llvmDialect) {
  using namespace mlir;

  auto uint64Ty = LLVM::LLVMType::getInt64Ty(llvmDialect);
  auto int32Ty = LLVM::LLVMType::getInt32Ty(llvmDialect);

  // fixme is it guaranteed for enum to have i32 type?
  return LLVM::LLVMType::getStructTy(uint64Ty, int32Ty, uint64Ty,
                                     uint64Ty.getPointerTo());
}
