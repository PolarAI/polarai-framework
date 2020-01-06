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

#ifndef ATHENA_LLVMTRANSLATOR_H
#define ATHENA_LLVMTRANSLATOR_H

#include <iostream>
#include <llvm/IR/IRBuilder.h>
#include <llvm/IR/Module.h>
#include <mlir/IR/Function.h>
#include <mlir/IR/Module.h>
#include <unordered_map>

// todo move this file to source dir. For now it is only used for testing.

namespace athena::backend::llvm {

/// Utility routines for lowering MLIR Graph dialect to LLVM IR
class LLVMTranslator {
private:
  ::llvm::LLVMContext& mLLVMContext;
  mlir::OwningModuleRef& mMLIRModule;
  ::llvm::Module& mLLVMModule;
  ::llvm::IRBuilder<> mBuilder;
  std::unordered_map<size_t, ::llvm::Function*> mNodeFunctionsMap;
  std::unordered_map<std::string, ::llvm::Function*> mGraphFunctionsMap;

  /// Returns function that represents a Node. If none exists, a new function
  /// is created.
  ///
  /// \param nodeId is an identifier of Node.
  /// \param nodeName is a unique name of Node.
  /// \param clusterId is an identifier of a cluster, where Node resides.
  /// \param graphFunc is a pointer to function that represents a Graph where
  /// the Node resides.
  /// \return a pointer to a function.
  ::llvm::Function* getOrCreateNodeFunction(size_t nodeId,
                                            ::llvm::StringRef nodeName,
                                            size_t clusterId,
                                            ::llvm::Function* graphFunc);

  /// \param graphName is a unique name of a Graph.
  /// \return a pointer to a function that represents a Graph. If none exists,
  /// a new function is created.
  ::llvm::Function* getOrCreateGraphFunction(const std::string& graphName);

  /// \param nodeFunction is a pointer to a function that represents a Node,
  /// where this tensor is required.
  /// \param tensorAddr is a virtual address of a Tensor.
  /// \return a pointer to a Tensor object.
  ::llvm::Value* getTensorPtr(::llvm::Function* nodeFunction,
                              size_t tensorAddr);

  /// \param nodeFunction is a pointer to a function that represents a Node,
  /// where this Device is required.
  /// \param nodeId is an identifier of Node.
  /// \return a pointer to a Device to dispatch current Node to.
  ::llvm::Value* getDeviceForNode(::llvm::Function* nodeFunction,
                                  size_t nodeId);

  /// \param nodeFunction is a pointer to a function that represents a Node,
  /// where Allocator is required.
  /// \return a pointer to the Allocator.
  ::llvm::Value* getAllocator(::llvm::Function* nodeFunction);

  /// \param mlirOp is an operation to extract options from.
  /// \return a pointer to a value of options.
  ::llvm::Value* getOptionsValue(mlir::Operation& mlirOp);

  /// Generate call to allocate.
  ///
  /// \param nodeFunction is a pointer to a function that represents a Node,
  /// where allocation is required.
  /// \param tensorAddr is a virtual address of tensor to allocate memory for.
  /// \param nodeId is an identifier of a Node where allocation is required.
  void addAlloca(::llvm::Function* nodeFunction, size_t tensorAddr,
                 size_t nodeId);

  /// Generate call to builtin.
  ///
  /// \param nodeFunction is a pointer to a function that represents a Node,
  /// where allocation is required.
  /// \param mlirOp is a MLIR operation for builtin.
  /// \param nodeId is an identifier of a Node where allocation is required.
  void addGenericBuiltin(mlir::Operation& mlirOp,
                         ::llvm::Function* nodeFunction, size_t nodeId);

public:
  /// Constructs translator object.
  ///
  /// \param mlirModule is a reference to MLIR module containing Graph IR.
  /// \param llvmModule is a reference to output LLVM IR module.
  LLVMTranslator(mlir::OwningModuleRef& mlirModule, ::llvm::Module& llvmModule)
      : mLLVMContext(llvmModule.getContext()), mMLIRModule(mlirModule),
        mLLVMModule(llvmModule), mBuilder(::llvm::IRBuilder(mLLVMContext)){};

  /// Performs translation.
  void translate();
};

} // namespace athena::backend::llvm

#endif // ATHENA_LLVMTRANSLATOR_H
