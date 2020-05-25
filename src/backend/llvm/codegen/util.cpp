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
::llvm::Constant* getFPConstant<float>(::llvm::LLVMContext& ctx, float value) {
  return ::llvm::ConstantFP::get(::llvm::Type::getFloatTy(ctx), value);
}

template <>
::llvm::Constant* getFPConstant<double>(::llvm::LLVMContext& ctx,
                                        double value) {
  return ::llvm::ConstantFP::get(::llvm::Type::getDoubleTy(ctx), value);
}

template <typename T>
void generateStandardBuiltinCall(
    const std::string& name, LLVMGenerator* generator, ::llvm::LLVMContext& ctx,
    ::llvm::Module& module, ::llvm::IRBuilder<>& builder,
    core::internal::TensorInternal& a, core::internal::TensorInternal& b,
    core::internal::TensorInternal& c) {
  ::llvm::Function* calledFunction =
      generator->findLLVMFunction(Mangler::getMangledName<T>(name));

  if (!calledFunction) {
    utils::FatalError(utils::ATH_FATAL_OTHER, "Unknown function referenced");
  }

  std::vector<::llvm::Value*> ArgsV;

  ::llvm::Constant* device = ::llvm::ConstantInt::get(
      ::llvm::Type::getInt64Ty(ctx),
      reinterpret_cast<size_t>(generator->getPreferredDevice(name)));
  ArgsV.push_back(device);
  ::llvm::Constant* allocator = ::llvm::ConstantInt::get(
      ::llvm::Type::getInt64Ty(ctx),
      reinterpret_cast<size_t>(&generator->getAllocator()));
  ArgsV.push_back(allocator);
  ::llvm::Constant* aTensor = ::llvm::ConstantInt::get(
      ::llvm::Type::getInt64Ty(ctx), reinterpret_cast<size_t>(&a));
  ArgsV.push_back(aTensor);
  ::llvm::Constant* bTensor = ::llvm::ConstantInt::get(
      ::llvm::Type::getInt64Ty(ctx), reinterpret_cast<size_t>(&b));
  ArgsV.push_back(bTensor);
  ::llvm::Constant* cTensor = ::llvm::ConstantInt::get(
      ::llvm::Type::getInt64Ty(ctx), reinterpret_cast<size_t>(&c));
  ArgsV.push_back(cTensor);
  builder.CreateCall(calledFunction, ArgsV);
}

template <typename T>
void generateStandardBuiltinCall(
    const std::string& name, LLVMGenerator* generator, ::llvm::LLVMContext& ctx,
    ::llvm::Module& module, ::llvm::IRBuilder<>& builder,
    core::internal::TensorInternal& a, uint64_t scaleA,
    core::internal::TensorInternal& b, uint64_t scaleB,
    core::internal::TensorInternal& c) {
  auto realScaleA = *reinterpret_cast<T*>(&scaleA);
  auto realScaleB = *reinterpret_cast<T*>(&scaleB);

  ::llvm::Function* calledFunction =
      generator->findLLVMFunction(Mangler::getMangledName<T>(name));

  if (!calledFunction) {
    utils::FatalError(utils::ATH_FATAL_OTHER, "Unknown function referenced");
  }

  std::vector<::llvm::Value*> ArgsV;

  ::llvm::Constant* device = ::llvm::ConstantInt::get(
      ::llvm::Type::getInt64Ty(ctx),
      reinterpret_cast<size_t>(generator->getPreferredDevice(name)));
  ArgsV.push_back(device);
  ::llvm::Constant* allocator = ::llvm::ConstantInt::get(
      ::llvm::Type::getInt64Ty(ctx),
      reinterpret_cast<size_t>(&generator->getAllocator()));
  ArgsV.push_back(allocator);
  ::llvm::Constant* aTensor = ::llvm::ConstantInt::get(
      ::llvm::Type::getInt64Ty(ctx), reinterpret_cast<size_t>(&a));
  ArgsV.push_back(aTensor);
  ::llvm::Constant* scaleAConst = getFPConstant<T>(ctx, realScaleA);
  ArgsV.push_back(scaleAConst);
  ::llvm::Constant* bTensor = ::llvm::ConstantInt::get(
      ::llvm::Type::getInt64Ty(ctx), reinterpret_cast<size_t>(&b));
  ArgsV.push_back(bTensor);
  ::llvm::Constant* scaleBConst = getFPConstant<T>(ctx, realScaleB);
  ArgsV.push_back(scaleBConst);
  ::llvm::Constant* cTensor = ::llvm::ConstantInt::get(
      ::llvm::Type::getInt64Ty(ctx), reinterpret_cast<size_t>(&c));
  ArgsV.push_back(cTensor);
  builder.CreateCall(calledFunction, ArgsV);
}

template <typename T>
void generateStandardBuiltinCall(
    const std::string& name, LLVMGenerator* generator, ::llvm::LLVMContext& ctx,
    ::llvm::Module& module, ::llvm::IRBuilder<>& builder, void* opts,
    core::internal::TensorInternal& a, core::internal::TensorInternal& b,
    core::internal::TensorInternal& c) {
  ::llvm::Function* calledFunction =
      generator->findLLVMFunction(Mangler::getMangledName<T>(name));

  if (!calledFunction) {
    utils::FatalError(utils::ATH_FATAL_OTHER, "Unknown function referenced");
  }

  std::vector<::llvm::Value*> ArgsV;

  ::llvm::Constant* device = ::llvm::ConstantInt::get(
      ::llvm::Type::getInt64Ty(ctx),
      reinterpret_cast<size_t>(generator->getPreferredDevice(name)));
  ArgsV.push_back(device);
  ::llvm::Constant* allocator = ::llvm::ConstantInt::get(
      ::llvm::Type::getInt64Ty(ctx),
      reinterpret_cast<size_t>(&generator->getAllocator()));
  ArgsV.push_back(allocator);
  ::llvm::Constant* optsArg = ::llvm::ConstantInt::get(
      ::llvm::Type::getInt64Ty(ctx), reinterpret_cast<size_t>(opts));
  ArgsV.push_back(optsArg);
  ::llvm::Constant* aTensor = ::llvm::ConstantInt::get(
      ::llvm::Type::getInt64Ty(ctx), reinterpret_cast<size_t>(&a));
  ArgsV.push_back(aTensor);
  ::llvm::Constant* bTensor = ::llvm::ConstantInt::get(
      ::llvm::Type::getInt64Ty(ctx), reinterpret_cast<size_t>(&b));
  ArgsV.push_back(bTensor);
  ::llvm::Constant* cTensor = ::llvm::ConstantInt::get(
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
void registerStandardBuiltin<BuiltinThreeTensorArgs>(const std::string& name,
                                                     LLVMGenerator* generator) {
  BuiltinThreeTensorArgs f =
      [generator, name](::llvm::LLVMContext& ctx, ::llvm::Module& module,
                        ::llvm::IRBuilder<>& builder,
                        core::internal::TensorInternal& a,
                        core::internal::TensorInternal& b,
                        core::internal::TensorInternal& c) {
        if (a.getDataType() == core::DataType::FLOAT) {
          generateStandardBuiltinCall<float>(name, generator, ctx, module,
                                             builder, a, b, c);
        } else if (a.getDataType() == core::DataType::DOUBLE) {
          generateStandardBuiltinCall<double>(name, generator, ctx, module,
                                              builder, a, b, c);
        } else {
          new utils::FatalError(utils::ATH_FATAL_OTHER, "Unsupported type");
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
void registerStandardBuiltin<BuiltinTATATArgs>(const std::string& name,
                                               LLVMGenerator* generator) {
  std::function<void(::llvm::LLVMContext&, ::llvm::Module&,
                     ::llvm::IRBuilder<>&, core::internal::TensorInternal&,
                     uint64_t&, core::internal::TensorInternal&, uint64_t&,
                     core::internal::TensorInternal&)>
      f = [generator, name](::llvm::LLVMContext& ctx, ::llvm::Module& module,
                            ::llvm::IRBuilder<>& builder,
                            core::internal::TensorInternal& a, uint64_t scaleA,
                            core::internal::TensorInternal& b, uint64_t scaleB,
                            core::internal::TensorInternal& c) {
        if (a.getDataType() == core::DataType::FLOAT) {
          generateStandardBuiltinCall<float>(name, generator, ctx, module,
                                             builder, a, scaleA, b, scaleB, c);
        } else if (a.getDataType() == core::DataType::DOUBLE) {
          generateStandardBuiltinCall<double>(name, generator, ctx, module,
                                              builder, a, scaleA, b, scaleB, c);
        } else {
          new utils::FatalError(utils::ATH_FATAL_OTHER, "Unsupported type");
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
    const std::string& name, LLVMGenerator* generator) {
  std::function<void(
      ::llvm::LLVMContext&, ::llvm::Module&, ::llvm::IRBuilder<>&, void*&,
      core::internal::TensorInternal&, core::internal::TensorInternal&,
      core::internal::TensorInternal&)>
      f = [generator, name](::llvm::LLVMContext& ctx, ::llvm::Module& module,
                            ::llvm::IRBuilder<>& builder, void*& opts,
                            core::internal::TensorInternal& a,
                            core::internal::TensorInternal& b,
                            core::internal::TensorInternal& c) {
        if (a.getDataType() == core::DataType::FLOAT) {
          generateStandardBuiltinCall<float>(name, generator, ctx, module,
                                             builder, opts, a, b, c);
        } else if (a.getDataType() == core::DataType::DOUBLE) {
          generateStandardBuiltinCall<double>(name, generator, ctx, module,
                                              builder, opts, a, b, c);
        } else {
          new utils::FatalError(utils::ATH_FATAL_OTHER, "Unsupported type");
        }
      };

  generator->registerFunctor(name, f);
}

} // namespace athena::backend::llvm::codegen