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

#include <athena/backend/llvm/mlir/LLVMTranslator.h>
#include <unordered_map>
#include <utility>

namespace athena::backend::llvm {

// todo replace with real mangler
static std::string getTypePostfix(const std::string& type) {
  if (type == "float") {
    return "_f";
  }
  return "_d";
}

static std::string getMangledName(const std::string& name, std::string type) {
  return "athn_" + name + getTypePostfix(type);
}

void LLVMTranslator::translate() {
  for (auto& funcCandidate : *mMLIRModule) {
    // Filter out functions only
    if (funcCandidate.getName().getStringRef() != "func") {
      continue;
    }

    auto func = mlir::dyn_cast<mlir::FuncOp>(funcCandidate);
    for (auto& block : func) {
      auto* graphFunc = getOrCreateGraphFunction(func.getName().str());

      for (auto& op : block) {
        if (op.getName().getStringRef() == "graph.return")
          continue;
        size_t nodeId =
            static_cast<size_t>(*op.getAttrOfType<mlir::IntegerAttr>("node_id")
                                     .getValue()
                                     .getRawData());
        size_t clusterId =
            static_cast<size_t>(*op.getAttrOfType<mlir::IntegerAttr>("node_id")
                                     .getValue()
                                     .getRawData());
        auto* nodeFunc = getOrCreateNodeFunction(
            nodeId, op.getAttrOfType<mlir::StringAttr>("node_name").getValue(),
            clusterId, graphFunc);
        // todo(alexbatashev): Special treatment for call, memlock(release), etc
        if (op.getName().getStringRef() == "graph.alloca") {
          size_t tensorAddr = static_cast<size_t>(
              *op.getAttrOfType<mlir::IntegerAttr>("tensor_addr")
                   .getValue()
                   .getRawData());
          addAlloca(nodeFunc, tensorAddr, nodeId);
        } else {
          addGenericBuiltin(op, nodeFunc, nodeId);
        }
      }
    }
  }

  for (auto& funcPair : mNodeFunctionsMap) {
    mBuilder.SetInsertPoint(&funcPair.second->getEntryBlock());
    mBuilder.CreateRet(nullptr);
  }

  for (auto& funcPair : mGraphFunctionsMap) {
    mBuilder.SetInsertPoint(&funcPair.second->getEntryBlock());
    mBuilder.CreateRet(nullptr);
  }
}
::llvm::Function* LLVMTranslator::getOrCreateNodeFunction(
    size_t nodeId, ::llvm::StringRef nodeName, size_t clusterId,
    ::llvm::Function* graphFunc) {
  if (mNodeFunctionsMap.count(nodeId)) {
    return mNodeFunctionsMap[nodeId];
  }
  ::llvm::FunctionType* FT = ::llvm::FunctionType::get(
      ::llvm::Type::getVoidTy(mLLVMContext), {}, false);
  auto* nodeFunction = ::llvm::Function::Create(
      FT, ::llvm::Function::ExternalLinkage, nodeName, mLLVMModule);
  ::llvm::BasicBlock::Create(mLLVMContext, "entry", nodeFunction);
  mBuilder.SetInsertPoint(&graphFunc->getEntryBlock());
  mBuilder.CreateCall(nodeFunction);

  auto* nodeIdMetadata = ::llvm::MDNode::get(
      mLLVMContext, ::llvm::ConstantAsMetadata::get(::llvm::ConstantInt::get(
                        mLLVMContext, ::llvm::APInt(64, nodeId, false))));
  auto* nodeNameMetadata = ::llvm::MDNode::get(
      mLLVMContext, ::llvm::MDString::get(mLLVMContext, nodeName));
  auto* clusterIdMetadata = ::llvm::MDNode::get(
      mLLVMContext, ::llvm::ConstantAsMetadata::get(::llvm::ConstantInt::get(
                        mLLVMContext, ::llvm::APInt(64, clusterId, false))));
  nodeFunction->addMetadata("node_id", *nodeIdMetadata);
  nodeFunction->addMetadata("node_name", *nodeNameMetadata);
  nodeFunction->addMetadata("cluster_id", *clusterIdMetadata);

  mNodeFunctionsMap[nodeId] = nodeFunction;
  return nodeFunction;
}
::llvm::Function*
LLVMTranslator::getOrCreateGraphFunction(const std::string& graphName) {
  if (mGraphFunctionsMap.count(graphName)) {
    return mGraphFunctionsMap[graphName];
  }
  ::llvm::FunctionType* FT = ::llvm::FunctionType::get(
      ::llvm::Type::getVoidTy(mLLVMContext), {}, false);
  auto* graphFunction =
      ::llvm::Function::Create(FT, ::llvm::Function::ExternalLinkage,
                               "evaluate" + graphName, mLLVMModule);
  ::llvm::BasicBlock::Create(mLLVMContext, "entry", graphFunction);
  mGraphFunctionsMap[graphName] = graphFunction;
  return graphFunction;
}
void LLVMTranslator::addAlloca(::llvm::Function* nodeFunction,
                               size_t tensorAddr, size_t nodeId) {
  auto* devicePtr = getDeviceForNode(nodeFunction, nodeId);
  auto* allocatorPtr = getAllocator(nodeFunction);
  auto* tensorPtr = getTensorPtr(nodeFunction, tensorAddr);
  ::llvm::Function* func = mLLVMModule.getFunction("allocate");
  if (!func) {
    std::vector<::llvm::Type*> args(3, ::llvm::Type::getInt64Ty(mLLVMContext));
    ::llvm::FunctionType* FT = ::llvm::FunctionType::get(
        ::llvm::Type::getVoidTy(mLLVMContext), args, false);
    func = ::llvm::Function::Create(FT, ::llvm::Function::ExternalLinkage,
                                    "allocate", mLLVMModule);
  }

  mBuilder.SetInsertPoint(&nodeFunction->getEntryBlock());
  mBuilder.CreateCall(func, {devicePtr, allocatorPtr, tensorPtr});
}
::llvm::Value* LLVMTranslator::getTensorPtr(::llvm::Function* nodeFunction,
                                            size_t tensorAddr) {
  ::llvm::Function* func = mLLVMModule.getFunction("getTensorPtr");
  if (!func) {
    ::llvm::FunctionType* FT = ::llvm::FunctionType::get(
        ::llvm::Type::getInt64PtrTy(mLLVMContext),
        {::llvm::Type::getInt64Ty(mLLVMContext)}, false);
    func = ::llvm::Function::Create(FT, ::llvm::Function::ExternalLinkage,
                                    "getTensorPtr", mLLVMModule);
  }
  mBuilder.SetInsertPoint(&nodeFunction->getEntryBlock());
  auto* constTensorAddr = ::llvm::ConstantInt::get(
      ::llvm::Type::getInt64Ty(mLLVMContext), tensorAddr);
  return mBuilder.CreateCall(func, {constTensorAddr});
}

::llvm::Value* LLVMTranslator::getDeviceForNode(::llvm::Function* nodeFunction,
                                                size_t nodeId) {
  ::llvm::Function* func = mLLVMModule.getFunction("getDeviceForNode");
  if (!func) {
    ::llvm::FunctionType* FT = ::llvm::FunctionType::get(
        ::llvm::Type::getInt64PtrTy(mLLVMContext),
        {::llvm::Type::getInt64Ty(mLLVMContext)}, false);
    func = ::llvm::Function::Create(FT, ::llvm::Function::ExternalLinkage,
                                    "getDeviceForNode", mLLVMModule);
  }
  mBuilder.SetInsertPoint(&nodeFunction->getEntryBlock());
  auto* constTensorAddr =
      ::llvm::ConstantInt::get(::llvm::Type::getInt64Ty(mLLVMContext), nodeId);
  return mBuilder.CreateCall(func, {constTensorAddr});
}

::llvm::Value* LLVMTranslator::getAllocator(::llvm::Function* nodeFunction) {
  ::llvm::Function* func = mLLVMModule.getFunction("getAllocator");
  if (!func) {
    ::llvm::FunctionType* FT = ::llvm::FunctionType::get(
        ::llvm::Type::getInt64PtrTy(mLLVMContext), {}, false);
    func = ::llvm::Function::Create(FT, ::llvm::Function::ExternalLinkage,
                                    "getAllocator", mLLVMModule);
  }
  mBuilder.SetInsertPoint(&nodeFunction->getEntryBlock());
  return mBuilder.CreateCall(func);
}

::llvm::Value* LLVMTranslator::getOptionsValue(mlir::Operation& mlirOp) {
  // todo(alexbatashev) implementation
  // Many builtins make use of options. Implementing this feature requires
  // changes to graph compiler, operations infra, which is out of scope of this
  // patch.
  return nullptr;
}

void LLVMTranslator::addGenericBuiltin(mlir::Operation& mlirOp,
                                       ::llvm::Function* nodeFunction,
                                       size_t nodeId) {
  auto args = mlirOp.getOperands();
  std::vector<size_t> tensorAddrs;
  tensorAddrs.reserve(args.size());
  for (const auto& arg : args) {
    auto* defOp = arg.getDefiningOp();
    size_t tensorAddr = static_cast<size_t>(
        *defOp->getAttrOfType<mlir::IntegerAttr>("tensor_addr")
             .getValue()
             .getRawData());
    tensorAddrs.push_back(tensorAddr);
  }
  std::vector<::llvm::Value*> argValues;
  argValues.reserve(args.size() + 2);
  argValues.push_back(getDeviceForNode(nodeFunction, nodeId));
  argValues.push_back(getAllocator(nodeFunction));
  std::transform(tensorAddrs.begin(), tensorAddrs.end(),
                 std::back_inserter(argValues),
                 [&](size_t addr) { return getTensorPtr(nodeFunction, addr); });

  auto* options = getOptionsValue(mlirOp);
  if (options)
    argValues.push_back(options);

  // todo handle types
  std::string mangledName =
      getMangledName(mlirOp.getName().getStringRef().str().substr(6), "float");

  ::llvm::Function* func = mLLVMModule.getFunction(mangledName);
  if (!func) {
    std::vector<::llvm::Type*> argTypes(args.size(),
                                        ::llvm::Type::getInt64Ty(mLLVMContext));
    ::llvm::FunctionType* FT = ::llvm::FunctionType::get(
        ::llvm::Type::getVoidTy(mLLVMContext), {argTypes}, false);
    func = ::llvm::Function::Create(FT, ::llvm::Function::ExternalLinkage,
                                    mangledName, mLLVMModule);
  }
  mBuilder.SetInsertPoint(&nodeFunction->getEntryBlock());
  mBuilder.CreateCall(func, argValues);
}
} // namespace athena::backend::llvm