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

#include "ArgInfo.h"
#include "../utils/LaunchCommand.h"

#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Dialect/StandardOps/IR/Ops.h"
#include "mlir/IR/Types.h"

uint32_t getArgsCount(mlir::Operation* op) { return op->getNumOperands(); }

void fillArgDesc(mlir::Value argDescArray, mlir::Operation* op,
                 llvm::ArrayRef<mlir::Value> operands,
                 mlir::ConversionPatternRewriter& rewriter) {
  using namespace mlir;
  auto llvmDialect =
      op->getContext()->getRegisteredDialect<LLVM::LLVMDialect>();
  auto argDescTy = getArgDescType(llvmDialect);

  auto zero = rewriter.create<LLVM::ConstantOp>(
      op->getLoc(), LLVM::LLVMType::getInt32Ty(llvmDialect),
      rewriter.getIntegerAttr(rewriter.getIntegerType(32), 0));
  auto one = rewriter.create<LLVM::ConstantOp>(
      op->getLoc(), LLVM::LLVMType::getInt32Ty(llvmDialect),
      rewriter.getIntegerAttr(rewriter.getIntegerType(32), 1));
  auto two = rewriter.create<LLVM::ConstantOp>(
      op->getLoc(), LLVM::LLVMType::getInt32Ty(llvmDialect),
      rewriter.getIntegerAttr(rewriter.getIntegerType(32), 2));
  for (auto& arg : llvm::enumerate(operands)) {
    auto operandIndex = rewriter.create<LLVM::ConstantOp>(
        op->getLoc(), LLVM::LLVMType::getInt32Ty(llvmDialect),
        rewriter.getIntegerAttr(rewriter.getIntegerType(32), arg.index()));

    auto curArg = rewriter.create<LLVM::GEPOp>(
        op->getLoc(), argDescTy, argDescArray, ValueRange{zero, operandIndex});
    auto argPtr = rewriter.create<LLVM::GEPOp>(
        op->getLoc(), argDescTy.getStructElementType(1).getPointerTo(), curArg,
        ValueRange{zero, one});
    auto typePtr = rewriter.create<LLVM::GEPOp>(
        op->getLoc(), argDescTy.getStructElementType(2).getPointerTo(), curArg,
        ValueRange{zero, two});

    if (op->getOperand(arg.index()).getType().isa<mlir::RankedTensorType>()) {
      rewriter.create<LLVM::StoreOp>(op->getLoc(), arg.value(), argPtr);
      rewriter.create<LLVM::StoreOp>(op->getLoc(), zero, typePtr);
    } else {
      // fixme is it safe to assume that these types match onto C types?
      auto curArgType = arg.value().getType().cast<LLVM::LLVMType>();
      size_t byteWidth = 0;

      if (op->getOperand(arg.index()).getType().isInteger(32)) {
        byteWidth = sizeof(uint32_t);
      } else if (op->getOperand(arg.index()).getType().isInteger(64)) {
        byteWidth = sizeof(uint64_t);
      } else if (op->getOperand(arg.index()).getType().isF64()) {
        byteWidth = sizeof(double);
      } else if (op->getOperand(arg.index()).getType().isF32()) {
        byteWidth = sizeof(float);
      }

      // fixme byte is not always 8 bits.
      auto argData = rewriter.create<LLVM::AllocaOp>(
          op->getLoc(), curArgType.getPointerTo(), one, byteWidth * 8);
      rewriter.create<LLVM::StoreOp>(op->getLoc(), arg.value(), argData);
      auto bitcastArg = rewriter.create<LLVM::BitcastOp>(
          op->getLoc(), LLVM::LLVMType::getInt8Ty(llvmDialect).getPointerTo(),
          argData);
      rewriter.create<LLVM::StoreOp>(op->getLoc(), bitcastArg, argPtr);
      rewriter.create<LLVM::StoreOp>(op->getLoc(), one, typePtr);

      auto sizePtr = rewriter.create<LLVM::GEPOp>(
          op->getLoc(), argDescTy.getStructElementType(0).getPointerTo(),
          curArg, ValueRange{zero, zero});
      auto sizetType =
          LLVM::LLVMType::getIntNTy(llvmDialect, sizeof(size_t) * 8);
      auto sizeConst = rewriter.create<LLVM::ConstantOp>(
          op->getLoc(), sizetType,
          rewriter.getIntegerAttr(rewriter.getIntegerType(sizeof(size_t) * 8),
                                  byteWidth));
      rewriter.create<LLVM::StoreOp>(op->getLoc(), sizeConst, sizePtr);
    }
  }
}
