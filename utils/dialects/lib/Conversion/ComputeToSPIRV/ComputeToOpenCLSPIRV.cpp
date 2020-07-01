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

#include "Compute/ComputeOps.h"
#include "Conversion/ComputeToSPIRVPass.h"

#include "mlir/Conversion/GPUToSPIRV/ConvertGPUToSPIRV.h"
#include "mlir/Conversion/StandardToSPIRV/ConvertStandardToSPIRV.h"
#include "mlir/Dialect/SPIRV/SPIRVDialect.h"
#include "mlir/Dialect/SPIRV/SPIRVLowering.h"
#include "mlir/Dialect/SPIRV/SPIRVOps.h"
#include "mlir/IR/Module.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/DialectConversion.h"

using namespace mlir;

namespace {
class ComputeFuncOpLowering : public SPIRVOpLowering<compute::FuncOp> {
public:
  using SPIRVOpLowering<compute::FuncOp>::SPIRVOpLowering;

  LogicalResult
  matchAndRewrite(compute::FuncOp funcOp, ArrayRef<Value> operands,
                  ConversionPatternRewriter &rewriter) const override {
    // fixme check for kernel

    SmallVector<spirv::InterfaceVarABIAttr, 4> argABI;
    for (auto argIndex : llvm::seq<unsigned>(0, funcOp.getNumArguments())) {
      Optional<spirv::StorageClass> sc;
      if (funcOp.getArgument(argIndex).getType().isIntOrIndexOrFloat())
        sc = spirv::StorageClass::StorageBuffer;
      argABI.push_back(spirv::getInterfaceVarABIAttr(0, argIndex, sc,
                                                     rewriter.getContext()));
    }

    auto fnType = funcOp.getType();

    TypeConverter::SignatureConversion signatureConverter(
        fnType.getNumInputs());
    {
      for (auto argType : enumerate(funcOp.getType().getInputs())) {
        auto convertedType = typeConverter.convertType(argType.value());
        signatureConverter.addInputs(argType.index(), convertedType);
      }
    }

    auto newFuncOp = rewriter.create<spirv::FuncOp>(
        funcOp.getLoc(), funcOp.getName(),
        rewriter.getFunctionType(signatureConverter.getConvertedTypes(),
                                 llvm::None));
    for (const auto &namedAttr : funcOp.getAttrs()) {
      if (namedAttr.first == impl::getTypeAttrName() ||
          namedAttr.first == SymbolTable::getSymbolAttrName())
        continue;
      newFuncOp.setAttr(namedAttr.first, namedAttr.second);
    }
    rewriter.inlineRegionBefore(funcOp.getBody(), newFuncOp.getBody(),
                                newFuncOp.end());
    rewriter.applySignatureConversion(&newFuncOp.getBody(), signatureConverter);
    rewriter.eraseOp(funcOp);

    SmallVector<Attribute, 1> interfaceVars;
    rewriter.create<spirv::EntryPointOp>(funcOp.getLoc(),
                                         spirv::ExecutionModel::Kernel,
                                         newFuncOp, interfaceVars);

    return success();
  }
};

class ComputeModuleLowering : public SPIRVOpLowering<compute::ModuleOp> {
public:
  using SPIRVOpLowering<compute::ModuleOp>::SPIRVOpLowering;

  LogicalResult
  matchAndRewrite(compute::ModuleOp moduleOp, ArrayRef<Value> operands,
                  ConversionPatternRewriter &rewriter) const override;
};

class ComputeReturnLowering : public SPIRVOpLowering<compute::ReturnOp> {
public:
  using SPIRVOpLowering<compute::ReturnOp>::SPIRVOpLowering;

  LogicalResult
  matchAndRewrite(compute::ReturnOp moduleOp, ArrayRef<Value> operands,
                  ConversionPatternRewriter &rewriter) const override;
};

class ComputeToOpenCLSPIRVPass
    : public PassWrapper<ComputeToOpenCLSPIRVPass, OperationPass<ModuleOp>> {
public:
  void runOnOperation() override;
};
} // namespace

LogicalResult ComputeReturnLowering::matchAndRewrite(
    compute::ReturnOp returnOp, ArrayRef<Value> operands,
    ConversionPatternRewriter &rewriter) const {

  // todo adapt to use ReturnValue when compute dialect gets support for return

  rewriter.replaceOpWithNewOp<spirv::ReturnOp>(returnOp);

  return success();
}

void ComputeToOpenCLSPIRVPass::runOnOperation() {
  MLIRContext *context = &getContext();
  ModuleOp module = getOperation();

  SmallVector<Operation *, 1> computeModules;
  OpBuilder builder(context);

  module.walk([&builder, &computeModules](compute::ModuleOp moduleOp) {
    builder.setInsertionPoint(moduleOp.getOperation());
    computeModules.push_back(builder.clone(*moduleOp.getOperation()));
  });

  // fixme spir-v version and capabilities must be inferred from module.
  auto triple = spirv::VerCapExtAttr::get(
      spirv::Version::V_1_0, {spirv::Capability::Kernel},
      ArrayRef<spirv::Extension>(), context);
  auto targetAttr = spirv::TargetEnvAttr::get(
      triple, spirv::getDefaultResourceLimits(context));

  std::unique_ptr<ConversionTarget> target =
      spirv::SPIRVConversionTarget::get(targetAttr);

  SPIRVTypeConverter typeConverter(targetAttr);
  OwningRewritePatternList patterns;
  // Structured control flow
  populateGPUToSPIRVPatterns(context, typeConverter, patterns);
  populateStandardToSPIRVPatterns(context, typeConverter, patterns);
  populateComputeToOpenCLSPIRVPatterns(context, typeConverter, patterns);

  if (failed(applyFullConversion(computeModules, *target, patterns))) {
    return signalPassFailure();
  }
}

LogicalResult ComputeModuleLowering::matchAndRewrite(
    compute::ModuleOp moduleOp, ArrayRef<Value> operands,
    ConversionPatternRewriter &rewriter) const {
  auto spvModule = rewriter.create<spirv::ModuleOp>(
      moduleOp.getLoc(), spirv::AddressingModel::Logical,
      spirv::MemoryModel::OpenCL);

  // fixme for prototyping only. This info must be obtained from module.
  auto triple = spirv::VerCapExtAttr::get(
      spirv::Version::V_1_0, {spirv::Capability::Kernel},
      ArrayRef<spirv::Extension>(), moduleOp.getContext());
  spvModule.vce_tripleAttr(triple);

  // Move the region from the module op into the SPIR-V module.
  Region &spvModuleRegion = spvModule.body();
  rewriter.inlineRegionBefore(moduleOp.body(), spvModuleRegion,
                              spvModuleRegion.begin());
  // The spv.module build method adds a block with a terminator. Remove that
  // block. The terminator of the module op in the remaining block will be
  // legalized later.
  spvModuleRegion.back().erase();
  rewriter.eraseOp(moduleOp);
  return success();
}

namespace {
#include "ComputeToSPIRV.cpp.inc"
}

namespace mlir {

void populateComputeToOpenCLSPIRVPatterns(MLIRContext *context,
                                          SPIRVTypeConverter &typeConverter,
                                          OwningRewritePatternList &patterns) {
  populateWithGenerated(context, &patterns);
  patterns.insert<ComputeModuleLowering, ComputeFuncOpLowering,
                  ComputeReturnLowering>(context, typeConverter);
}
std::unique_ptr<OperationPass<ModuleOp>>
createConvertComputeToOpenCLSPIRVPass() {
  return std::make_unique<ComputeToOpenCLSPIRVPass>();
}
} // namespace mlir
