//===----------------------------------------------------------------------===//
// Copyright (c) 2020 PolarAI. All rights reserved.
//
// Licensed under MIT license.
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS, WITHOUT
// WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the
// License for the specific language governing permissions and limitations under
// the License.
//===----------------------------------------------------------------------===//

#include "../runtime/driver/RuntimeDriver.hpp"
#include "../runtime/host/HostDevice.hpp"
#include "GraphPartitionPlanner.hpp"
#include "allocators/LayerAllocator.hpp"
#include "jit/PolarJIT.hpp"

#include <Compute/ComputeDialect.h>
#include <PolarGraph/PolarGraphDialect.h>
#include <PolarRuntime/PolarRuntimeDialect.h>
#include <polarai/backend/generic/CodeGen.hpp>
#include <polarai/backend/generic/GenericExecutor.hpp>
#include <polarai/backend/generic/runtime/GraphHandle.hpp>
#include <polarai/core/Generator.hpp>
#include <polarai/core/graph/internal/GraphCompiler.hpp>
#include <polarai/utils/error/FatalError.hpp>

#include "mlir/IR/Builders.h"
#include "mlir/IR/Dialect.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/InitAllDialects.h"
#include "mlir/InitAllPasses.h"
#include "mlir/Parser.h"
#include "mlir/Pass/Pass.h"
#include "llvm/ExecutionEngine/ExecutionEngine.h"
#include "llvm/Support/Host.h"
#include "llvm/Support/InitLLVM.h"
#include "llvm/Support/TargetSelect.h"

#include <algorithm>
#include <memory>

using namespace polarai::core;

template <typename FuncT, typename T> static FuncT* func_cast(T x) {
  return reinterpret_cast<FuncT*>(static_cast<intptr_t>(x));
}

namespace polarai::backend::generic {

void GenericExecutor::addGraph(Graph& graph) {
  Generator generator;

  mlir::OpBuilder opBuilder(mJITCompiler->getContext());
  auto module = opBuilder.create<mlir::ModuleOp>(opBuilder.getUnknownLoc());
  mlir::OwningModuleRef ref(module);
  opBuilder.setInsertionPointToStart(module.getBody());
  populateCodeGenPatterns(generator, opBuilder);

  core::internal::GraphCompiler::compile(graph, generator);

  mJITCompiler->addModule(ref);
}

void GenericExecutor::evaluate(Graph& graph) {
  auto sym = mJITCompiler->lookupSymbol(graph.getName().getString());
  utils::polarai_assert((bool)sym, "Failed to find graph function. ",
                       "Did you forget to add Graph?");

  GraphHandle handle;
  handle.allocator = mAllocator;
  handle.devices.push_back(mDevices.front());
  // fixme host device must be loaded through runtime driver
  handle.devices.emplace_back(new HostDevice());

  auto& traversal = graph.traverse();

  auto ctxInternal = graph.getContext().internal();
  auto frontCluster = traversal.getClusters().front();
  for (const auto& node : frontCluster.content) {
    auto& nodeInternal = ctxInternal->get<AbstractNodeInternal>(node.nodeIndex);

    if (nodeInternal.getType() == NodeType::INPUT) {
      auto inpNode = ctxInternal->get<InputNodeInternal>(node.nodeIndex);
      auto loaderIdx = inpNode.getLoader();
      auto& loader = ctxInternal->get<AbstractLoaderInternal>(loaderIdx);
      // fixme do not use const cast
      handle.mLoaders[node.nodeIndex] =
          const_cast<AbstractLoaderInternal*>(&loader);
      handle.isHostNode.insert(node.nodeIndex);
    }
  }

  auto evaluateFunction = func_cast<void(void*)>(sym);
  evaluateFunction(&handle);
}

GenericExecutor::GenericExecutor(bool enableDebugOutput, FilterFunctionT filter)
    : mFilter(std::move(filter)) {
  mlir::registerAllDialects();
  mlir::registerAllPasses();

  mlir::registerDialect<mlir::polar_graph::PolarGraphDialect>();
  mlir::registerDialect<mlir::polar_rt::PolarRuntimeDialect>();
  mlir::registerDialect<mlir::compute::ComputeDialect>();

  ::llvm::InitializeAllTargets();
  ::llvm::InitializeAllTargetMCs();
  ::llvm::InitializeAllAsmPrinters();
  ::llvm::InitializeAllAsmParsers();

#ifdef DEBUG
  mJITCompiler = PolarJIT::createWithDebugging();
#else
  mJITCompiler = PolarJIT::create();
#endif
  if (!mJITCompiler) {
    new utils::FatalError(utils::ATH_FATAL_OTHER,
                          "Unable to create JIT compiler");
  }

  mAllocator = std::make_shared<LayerAllocator>();
  mRuntimeDriver = std::make_shared<RuntimeDriver>(enableDebugOutput);

  std::copy_if(mRuntimeDriver->getDeviceList().begin(),
               mRuntimeDriver->getDeviceList().end(),
               std::back_inserter(mDevices), mFilter);

  for (auto& dev : mDevices) {
    if (enableDebugOutput) {
      std::clog << "Registering " << dev->getDeviceName() << '\n';
    }
    mAllocator->registerDevice(*dev);
    mJITCompiler->registerDevice(dev);
  }
}

void GenericExecutor::addModule(std::string_view module) {
  auto moduleRef =
      mlir::parseSourceString(module.data(), mJITCompiler->getContext());

  mJITCompiler->addModule(moduleRef);
}

void GenericExecutor::execute(std::string_view name, void* userData) {
  auto sym = mJITCompiler->lookupSymbol(name.data());
  utils::polarai_assert((bool)sym, "Failed to find function.");

  auto evaluateFunction = func_cast<void(void*)>(sym);
  evaluateFunction(userData);
}

BackendAllocator& GenericExecutor::getAllocator() { return *mAllocator; }
std::shared_ptr<BackendAllocator> GenericExecutor::getAllocatorPtr() {
  return mAllocator;
}

void GenericExecutor::setAllocator(
    std::shared_ptr<BackendAllocator>& allocator) {
  mAllocator = std::move(allocator);
}

std::vector<std::shared_ptr<Device>>& GenericExecutor::getDevices() {
  return mDevices;
}
} // namespace polarai::backend::generic
