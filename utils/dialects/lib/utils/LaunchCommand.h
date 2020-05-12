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
getArgDescType(mlir::LLVM::LLVMDialect* llvmDialect) {
  using namespace mlir;
  // ArgDesc structure
  // fixme byte is not always 8 bits
  auto sizeTy = LLVM::LLVMType::getIntNTy(llvmDialect, sizeof(size_t) * 8);
  auto argTy = LLVM::LLVMType::getInt8Ty(llvmDialect).getPointerTo();
  // fixme is it guaranteed for enum to have i32 type?
  auto argTypeTy = LLVM::LLVMType::getInt32Ty(llvmDialect);

  return LLVM::LLVMType::getStructTy(sizeTy, argTy, argTypeTy);
}

inline mlir::LLVM::LLVMType
getLaunchCommandType(mlir::LLVM::LLVMDialect* llvmDialect) {
  using namespace mlir;

  LLVM::LLVMType argDescTy = getArgDescType(llvmDialect);
  auto kernelNameTy = LLVM::LLVMType::getInt8Ty(llvmDialect).getPointerTo();
  auto argsCountTy = LLVM::LLVMType::getInt32Ty(llvmDialect);
  auto argsTy = argDescTy.getPointerTo();
  auto workDimTy = LLVM::LLVMType::getIntNTy(llvmDialect, sizeof(size_t) * 8);
  auto dimSizeTy =
      LLVM::LLVMType::getIntNTy(llvmDialect, sizeof(size_t) * 8).getPointerTo();
  return LLVM::LLVMType::getStructTy(kernelNameTy, argsCountTy, argsTy,
                                     workDimTy, dimSizeTy, dimSizeTy);
}
