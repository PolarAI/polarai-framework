/*
 * Copyright (c) 2019 Athena. All rights reserved.
 * https://getathena.ml
 *
 * Licensed under MIT license.
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an “AS IS” BASIS, WITHOUT
 * WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the
 * License for the specific language governing permissions and limitations under
 * the License.
 */

#include "GraphPartitionPlanner.h"
#include "LLVMGenerator.h"
#include "allocators/LayerAllocator.h"
#include "jit/AthenaJIT.h"
#include "runtime/legacy_driver/runtime-driver.h"

#include <athena/backend/llvm/LLVMExecutor.h>
#include <athena/core/graph/internal/GraphCompiler.h>
#include <athena/utils/error/FatalError.h>

#include "llvm/ExecutionEngine/ExecutionEngine.h"
#include "llvm/Support/Host.h"
#include <algorithm>
#include <athena/utils/error/FatalError.h>

namespace athena::backend::llvm {

void LLVMExecutor::setGraph(athena::core::Graph& graph) {
  auto modules = compileGraph(graph);

  // At the moment compileGraph method always returns exactly 1 module.
  // That may change in future when we decide to go with a more complex
  // structure of neural networks.
  for (auto& module : modules) {
    auto err = mJITCompiler->addModule(module);
    if (err) {
      utils::FatalError(utils::ATH_FATAL_OTHER, "Error adding module to JIT");
    }
  }

  // Prepare runtime library
  auto& tmp = mRuntimeDriver->getModules();
  for (auto& module : tmp) {
    module->setDataLayout(mJITCompiler->getDataLayout()); // fixme hack
    auto err = mJITCompiler->addModule(module);
    if (err) {
      new utils::FatalError(utils::ATH_FATAL_OTHER, "Unable to add module");
    }
  }
}

void LLVMExecutor::evaluate() {
  auto sym = mJITCompiler->lookup("evaluateGraph");
  utils::athena_assert((bool)sym, "Failed to find evaluateGraph function. ",
                       "Did you forget to set Graph?");

  auto evaluateFunction = (void (*)())(intptr_t)sym.get().getAddress();
  evaluateFunction();
}

void LLVMExecutor::optimizeGraph() {
  auto sym = mJITCompiler->lookup("optimizeGraph");
  utils::athena_assert((bool)sym, "Failed to find optimizeGraph function. ",
                       "Did you forget to set Graph?");

  auto optimizeFunction = (void (*)())(intptr_t)sym.get().getAddress();
  optimizeFunction();
}

LLVMExecutor::LLVMExecutor() : mJITCompiler(AthenaJIT::create()) {
  if (!mJITCompiler) {
    new utils::FatalError(utils::ATH_FATAL_OTHER,
                          "Unable to create JIT compiler");
  }

  mRuntimeDriver =
      std::make_unique<LegacyRuntimeDriver>(mJITCompiler->getContext());

  // TODO better RT lib handling
  auto libName = std::getenv("ATHENA_RT_LIBRARY");
  mRuntimeDriver->load(libName);
  utils::athena_assert(mRuntimeDriver->isLoaded(), "Failed to load runtime.");

  mAllocator = std::make_unique<LayerAllocator>();
}

llvm::BackendAllocator& LLVMExecutor::getAllocator() { return *mAllocator; }
void LLVMExecutor::setAllocator(std::unique_ptr<BackendAllocator>& allocator) {
  mAllocator = std::move(allocator);
}

std::vector<std::unique_ptr<::llvm::Module>>
LLVMExecutor::compileGraph(athena::core::Graph& graph) {
  auto llvmModule = std::make_unique<::llvm::Module>(
      graph.getName().getString(), mJITCompiler->getContext());

  llvmModule->setDataLayout(mJITCompiler->getDataLayout());
  // TODO get real target triple
  llvmModule->setTargetTriple(::llvm::sys::getDefaultTargetTriple());

  GraphPartitionPlanner planner(graph);
  // todo do actual partitioning
  DeviceContainer devices =
      planner.getPartitionedDevices(mRuntimeDriver->getAvailableDevices());
  auto partitioning = planner.getGraphPartitioning();

  for (size_t idx = 0; idx < devices.count; idx++) {
    mAllocator->registerDevice(devices.devices[idx]);
  }

  mRuntimeDriver->initializeContext(devices);

  LLVMGenerator generator(mJITCompiler->getContext(), llvmModule, *mAllocator,
                          mRuntimeDriver->getModules(), partitioning);

  core::internal::GraphCompiler::compileForward(graph, generator);
  core::internal::GraphCompiler::compileBackward(graph, generator);

  std::vector<std::unique_ptr<::llvm::Module>> resultModules;
  resultModules.push_back(std::move(llvmModule));

  return resultModules;
}
} // namespace athena::backend::llvm
