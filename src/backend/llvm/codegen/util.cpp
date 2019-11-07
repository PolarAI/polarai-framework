/*
 * Copyright (c) 2019 Athena. All rights reserved.
 * https://getathena.ml
 *
 * Licensed under MIT license.
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS, WITHOUT
 * WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the
 * License for the specific language governing permissions and limitations under
 * the License.
 */

#include "Mangler.h"
#include "common.h"

namespace athena::backend::llvm::codegen {
template <>
::llvm::Constant *getFPConstant<float>(::llvm::LLVMContext &ctx, float value) {
    return ::llvm::ConstantFP::get(::llvm::Type::getFloatTy(ctx), value);
}

template <>
::llvm::Constant *getFPConstant<double>(::llvm::LLVMContext &ctx,
                                        double value) {
    return ::llvm::ConstantFP::get(::llvm::Type::getDoubleTy(ctx), value);
}

template <typename T>
void generateStandardBuiltinCall(const std::string &name,
                                 LLVMGenerator *generator,
                                 ::llvm::LLVMContext &ctx,
                                 ::llvm::Module &module,
                                 ::llvm::IRBuilder<> &builder,
                                 core::inner::Tensor &a,
                                 core::inner::Tensor &b,
                                 core::inner::Tensor &c) {
    ::llvm::Function *calledFunction =
        generator->findLLVMFunction(Mangler::getMangledName<T>(name));

    if (!calledFunction) {
        core::FatalError(core::ATH_FATAL_OTHER, "Unknown function referenced");
    }

    std::vector<::llvm::Value *> ArgsV;

    ::llvm::Constant *device = ::llvm::ConstantInt::get(
        ::llvm::Type::getInt64Ty(ctx),
        reinterpret_cast<size_t>(generator->getPreferredDevice(name)));
    ArgsV.push_back(device);
    ::llvm::Constant *allocator = ::llvm::ConstantInt::get(
        ::llvm::Type::getInt64Ty(ctx),
        reinterpret_cast<size_t>(&generator->getAllocator()));
    ArgsV.push_back(allocator);
    ::llvm::Constant *aTensor = ::llvm::ConstantInt::get(
        ::llvm::Type::getInt64Ty(ctx), reinterpret_cast<size_t>(&a));
    ArgsV.push_back(aTensor);
    ::llvm::Constant *bTensor = ::llvm::ConstantInt::get(
        ::llvm::Type::getInt64Ty(ctx), reinterpret_cast<size_t>(&b));
    ArgsV.push_back(bTensor);
    ::llvm::Constant *cTensor = ::llvm::ConstantInt::get(
        ::llvm::Type::getInt64Ty(ctx), reinterpret_cast<size_t>(&c));
    ArgsV.push_back(cTensor);
    builder.CreateCall(calledFunction, ArgsV);
}

template <typename T>
void generateStandardBuiltinCall(const std::string &name,
                                 LLVMGenerator *generator,
                                 ::llvm::LLVMContext &ctx,
                                 ::llvm::Module &module,
                                 ::llvm::IRBuilder<> &builder,
                                 core::inner::Tensor &a,
                                 uint64_t scaleA,
                                 core::inner::Tensor &b,
                                 uint64_t scaleB,
                                 core::inner::Tensor &c) {
    auto realScaleA = *reinterpret_cast<T *>(&scaleA);
    auto realScaleB = *reinterpret_cast<T *>(&scaleB);

    ::llvm::Function *calledFunction =
        generator->findLLVMFunction(Mangler::getMangledName<T>(name));

    if (!calledFunction) {
        core::FatalError(core::ATH_FATAL_OTHER, "Unknown function referenced");
    }

    std::vector<::llvm::Value *> ArgsV;

    ::llvm::Constant *device = ::llvm::ConstantInt::get(
        ::llvm::Type::getInt64Ty(ctx),
        reinterpret_cast<size_t>(generator->getPreferredDevice(name)));
    ArgsV.push_back(device);
    ::llvm::Constant *allocator = ::llvm::ConstantInt::get(
        ::llvm::Type::getInt64Ty(ctx),
        reinterpret_cast<size_t>(&generator->getAllocator()));
    ArgsV.push_back(allocator);
    ::llvm::Constant *aTensor = ::llvm::ConstantInt::get(
        ::llvm::Type::getInt64Ty(ctx), reinterpret_cast<size_t>(&a));
    ArgsV.push_back(aTensor);
    ::llvm::Constant *scaleAConst = getFPConstant<T>(ctx, realScaleA);
    ArgsV.push_back(scaleAConst);
    ::llvm::Constant *bTensor = ::llvm::ConstantInt::get(
        ::llvm::Type::getInt64Ty(ctx), reinterpret_cast<size_t>(&b));
    ArgsV.push_back(bTensor);
    ::llvm::Constant *scaleBConst = getFPConstant<T>(ctx, realScaleB);
    ArgsV.push_back(scaleBConst);
    ::llvm::Constant *cTensor = ::llvm::ConstantInt::get(
        ::llvm::Type::getInt64Ty(ctx), reinterpret_cast<size_t>(&c));
    ArgsV.push_back(cTensor);
    builder.CreateCall(calledFunction, ArgsV);
}

template <typename T>
void generateStandardBuiltinCall(const std::string &name,
                                 LLVMGenerator *generator,
                                 ::llvm::LLVMContext &ctx,
                                 ::llvm::Module &module,
                                 ::llvm::IRBuilder<> &builder,
                                 void *opts,
                                 core::inner::Tensor &a,
                                 core::inner::Tensor &b,
                                 core::inner::Tensor &c) {
    ::llvm::Function *calledFunction =
        generator->findLLVMFunction(Mangler::getMangledName<T>(name));

    if (!calledFunction) {
        core::FatalError(core::ATH_FATAL_OTHER, "Unknown function referenced");
    }

    std::vector<::llvm::Value *> ArgsV;

    ::llvm::Constant *device = ::llvm::ConstantInt::get(
        ::llvm::Type::getInt64Ty(ctx),
        reinterpret_cast<size_t>(generator->getPreferredDevice(name)));
    ArgsV.push_back(device);
    ::llvm::Constant *allocator = ::llvm::ConstantInt::get(
        ::llvm::Type::getInt64Ty(ctx),
        reinterpret_cast<size_t>(&generator->getAllocator()));
    ArgsV.push_back(allocator);
    ::llvm::Constant *optsArg = ::llvm::ConstantInt::get(
        ::llvm::Type::getInt64Ty(ctx), reinterpret_cast<size_t>(opts));
    ArgsV.push_back(optsArg);
    ::llvm::Constant *aTensor = ::llvm::ConstantInt::get(
        ::llvm::Type::getInt64Ty(ctx), reinterpret_cast<size_t>(&a));
    ArgsV.push_back(aTensor);
    ::llvm::Constant *bTensor = ::llvm::ConstantInt::get(
        ::llvm::Type::getInt64Ty(ctx), reinterpret_cast<size_t>(&b));
    ArgsV.push_back(bTensor);
    ::llvm::Constant *cTensor = ::llvm::ConstantInt::get(
        ::llvm::Type::getInt64Ty(ctx), reinterpret_cast<size_t>(&c));
    ArgsV.push_back(cTensor);
    builder.CreateCall(calledFunction, ArgsV);
}

/**
 * Register builtin with three Tensor args boilerplate routine
 * @param name Name of builtin
 * @param generator LLVMGenerator instance
 */
template <>
void registerStandardBuiltin<BuiltinThreeTensorArgs>(const std::string &name,
                                                     LLVMGenerator *generator) {
    BuiltinThreeTensorArgs f =
        [generator, name](::llvm::LLVMContext &ctx, ::llvm::Module &module,
                          ::llvm::IRBuilder<> &builder, core::inner::Tensor &a,
                          core::inner::Tensor &b, core::inner::Tensor &c) {
            if (a.getDataType() == core::DataType::FLOAT) {
                generateStandardBuiltinCall<float>(name, generator, ctx, module,
                                                   builder, a, b, c);
            } else if (a.getDataType() == core::DataType::DOUBLE) {
                generateStandardBuiltinCall<double>(name, generator, ctx,
                                                    module, builder, a, b, c);
            } else {
                new core::FatalError(core::ATH_FATAL_OTHER, "Unsupported type");
            }
        };
    generator->registerFunctor(name, f);
}

/**
 * Register builtin with T, A, T, A, T args
 * T == Tensor *,
 * A == arithmetic type (float, double)
 * @param name Name of builtin
 * @param generator LLVMGenerator instance
 */
template <>
void registerStandardBuiltin<BuiltinTATATArgs>(const std::string &name,
                                               LLVMGenerator *generator) {
    std::function<void(::llvm::LLVMContext &, ::llvm::Module &,
                       ::llvm::IRBuilder<> &, core::inner::Tensor &, uint64_t &,
                       core::inner::Tensor &, uint64_t &,
                       core::inner::Tensor &)>
        f = [generator, name](::llvm::LLVMContext &ctx, ::llvm::Module &module,
                              ::llvm::IRBuilder<> &builder,
                              core::inner::Tensor &a, uint64_t scaleA,
                              core::inner::Tensor &b, uint64_t scaleB,
                              core::inner::Tensor &c) {
            if (a.getDataType() == core::DataType::FLOAT) {
                generateStandardBuiltinCall<float>(name, generator, ctx, module,
                                                   builder, a, scaleA, b,
                                                   scaleB, c);
            } else if (a.getDataType() == core::DataType::DOUBLE) {
                generateStandardBuiltinCall<double>(name, generator, ctx,
                                                    module, builder, a, scaleA,
                                                    b, scaleB, c);
            } else {
                new core::FatalError(core::ATH_FATAL_OTHER, "Unsupported type");
            }
        };

    generator->registerFunctor(name, f);
}

/**
 * Register builtin with T, A, T, A, T args
 * T == Tensor *,
 * A == arithmetic type (float, double)
 * @param name Name of builtin
 * @param generator LLVMGenerator instance
 */
template <>
void registerStandardBuiltin<BuiltinThreeTensorWithOptsArgs>(
    const std::string &name, LLVMGenerator *generator) {
    std::function<void(::llvm::LLVMContext &, ::llvm::Module &,
                       ::llvm::IRBuilder<> &, void *&, core::inner::Tensor &,
                       core::inner::Tensor &, core::inner::Tensor &)>
        f = [generator, name](::llvm::LLVMContext &ctx, ::llvm::Module &module,
                              ::llvm::IRBuilder<> &builder, void *&opts,
                              core::inner::Tensor &a, core::inner::Tensor &b,
                              core::inner::Tensor &c) {
            if (a.getDataType() == core::DataType::FLOAT) {
                generateStandardBuiltinCall<float>(name, generator, ctx, module,
                                                   builder, opts, a, b, c);
            } else if (a.getDataType() == core::DataType::DOUBLE) {
                generateStandardBuiltinCall<double>(
                    name, generator, ctx, module, builder, opts, a, b, c);
            } else {
                new core::FatalError(core::ATH_FATAL_OTHER, "Unsupported type");
            }
        };

    generator->registerFunctor(name, f);
}

}