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

#include "GraphPartitionPlanner.h"
#include "allocators/LayerAllocator.h"
#include "jit/AthenaJIT.h"

#include <athena/backend/llvm/LLVMExecutor.h>
#include <athena/backend/llvm/CodeGen.h>
#include <athena/core/graph/internal/GraphCompiler.h>
#include <athena/core/Generator.h>
#include <athena/utils/error/FatalError.h>

#include "llvm/ExecutionEngine/ExecutionEngine.h"
#include "llvm/Support/Host.h"
#include "mlir/IR/Builders.h"

#include <algorithm>

using namespace athena::core;

template <typename FuncT, typename T> static FuncT* func_cast(T x) {
  return reinterpret_cast<FuncT*>(static_cast<intptr_t>(x));
}

namespace athena::backend::llvm {

void LLVMExecutor::addGraph(Graph& graph) {
  Generator generator;

  mlir::OpBuilder opBuilder(mJITCompiler->getContext());
  auto module = opBuilder.create<mlir::ModuleOp>(opBuilder.getUnknownLoc());
  mlir::OwningModuleRef ref(module);
  opBuilder.setInsertionPointToStart(module.getBody());
  populateCodeGenPatterns(generator, opBuilder);

  core::internal::GraphCompiler::compile(graph, generator);

  mJITCompiler->addModule(ref);
}

void LLVMExecutor::evaluate(Graph& graph) {
  auto sym = mJITCompiler->lookupSymbol(graph.getName().getString());
  utils::athena_assert((bool)sym, "Failed to find graph function. ",
                       "Did you forget to add Graph?");

  auto evaluateFunction = func_cast<void(void*)>(sym);
  evaluateFunction(nullptr);
}

LLVMExecutor::LLVMExecutor() : mJITCompiler(AthenaJIT::create()) {
  if (!mJITCompiler) {
    new utils::FatalError(utils::ATH_FATAL_OTHER,
                          "Unable to create JIT compiler");
  }

  mAllocator = std::make_unique<LayerAllocator>();
}

llvm::BackendAllocator& LLVMExecutor::getAllocator() { return *mAllocator; }
void LLVMExecutor::setAllocator(std::unique_ptr<BackendAllocator>& allocator) {
  mAllocator = std::move(allocator);
}
} // namespace athena::backend::llvm
