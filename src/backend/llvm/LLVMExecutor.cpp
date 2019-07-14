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

#include <athena/backend/llvm/LLVMExecutor.h>
#include <athena/backend/llvm/LLVMGenerator.h>
#include <athena/backend/llvm/runtime-driver/runtime-driver.h>
#include <athena/core/FatalError.h>
#include <athena/core/InputNode.h>
#include <athena/core/Node.h>
#include <athena/core/inner/GlobalTables.h>
#include <athena/core/inner/InnerFunctions.h>

#include "llvm/ExecutionEngine/ExecutionEngine.h"
#include "llvm/Support/TargetSelect.h"

#include <cassert>

namespace athena::backend::llvm {

void LLVMExecutor::prepare(athena::core::Graph &graph) {
    mMainModule = std::make_unique<::llvm::Module>("AthenaMain",
                                                   mJITCompiler->getContext());
    mMainModule->setDataLayout(mJITCompiler->getDataLayout());
    mMainModule->setTargetTriple(::llvm::sys::getDefaultTargetTriple());

    // todo consider initializing some optimizers

    ::llvm::FunctionType *FT = ::llvm::FunctionType::get(
        ::llvm::Type::getVoidTy(mJITCompiler->getContext()), false);
    ::llvm::Function::Create(FT, ::llvm::Function::ExternalLinkage, "jitmain",
                             *mMainModule);

    LLVMGenerator generator(mJITCompiler->getContext(), mMainModule,
                            *mAllocator, mRuntimeDriver->getModules());

    mGraphTraversal = graph.traverse();

    for (auto &cluster : mGraphTraversal.getClusters()) {
        auto &inputNodes = cluster.get<core::InputNode>();
        for (auto &nodeDeps : inputNodes) {
            auto &inputNode = static_cast<core::InputNode &>(
                *core::inner::getNodeTable()[nodeDeps.nodeIndex]);
            generator.openNode(inputNode.getName());
            generator.generate("allocate",
                               core::inner::getTensorFromNode(inputNode));
            // todo generate code for loader
            generator.generateLoad(inputNode.getLoader(),
                                   core::inner::getTensorFromNode(inputNode));
            generator.closeNode();
        }

        auto &actionNodes = cluster.get<core::Node>();
        for (auto &nodeDeps : actionNodes) {
            std::vector<core::inner::Tensor *> preparedTensors;
            for (auto &input : nodeDeps.input) {
                auto *node = core::inner::getNodeTable()[input.nodeIndex];
                preparedTensors.push_back(
                    &core::inner::getTensorFromNode(*node));
            }
            auto &node = static_cast<core::Node &>(
                *core::inner::getNodeTable()[nodeDeps.nodeIndex]);
            generator.openNode(node.getName());
            generator.generate("allocate",
                               core::inner::getTensorFromNode(node));
            preparedTensors.push_back(&core::inner::getTensorFromNode(node));
            // todo lock tensors in memory
            node.getOperation().gen(generator, preparedTensors);
            // todo unlock tensors in memory

            for (size_t argNo = 0; argNo < nodeDeps.input.size(); argNo++) {
                // todo check for frozen nodes
                auto derivativeTensor = node.getOperation().getDerivativeTensor(
                    preparedTensors, argNo);
                core::inner::addDerivativeTensor(node, *derivativeTensor);
                preparedTensors.pop_back();
                preparedTensors.push_back(derivativeTensor);
                generator.generate("allocate", *derivativeTensor);
                node.getOperation().genDerivative(generator, preparedTensors,
                                                  argNo);
            }

            generator.closeNode();
        }
    }

    auto builder = generator.getBuilder();

    builder.CreateRetVoid();

    for (auto &module : mRuntimeDriver->getModules()) {
        module->setDataLayout(mJITCompiler->getDataLayout());
        auto err = mJITCompiler->addModule(module);
        if (err) {
            new core::FatalError(1, "Unable to add module");
        }
    }
}

void LLVMExecutor::execute() {
    auto err = mJITCompiler->addModule(mMainModule);
    if (err) {
        core::FatalError(1, "Error adding module to JIT");
    }

    auto sym = mJITCompiler->lookup("jitmain");
#ifdef DEBUG
    assert(sym && "Failed to find jitmain function");
#endif

    auto mainFunction = (void (*)())(intptr_t)sym.get().getAddress();
    mainFunction();
}

LLVMExecutor::LLVMExecutor() : mJITCompiler(AthenaJIT::create()) {
    if (!mJITCompiler) {
        new core::FatalError(1, "Unable to create JIT compiler");
    }

    mRuntimeDriver =
        std::make_unique<RuntimeDriver>(mJITCompiler->getContext());

    auto libName = std::getenv("ATHENA_RT_LIBRARY");
    mRuntimeDriver->load(libName);
#ifdef DEBUG
    assert(mRuntimeDriver->isLoaded());
#endif
}

std::unique_ptr<core::Allocator> &LLVMExecutor::getAllocator() {
    return mAllocator;
}
void LLVMExecutor::setAllocator(std::unique_ptr<core::Allocator> &allocator) {
    mAllocator = std::move(allocator);
}

}  // namespace athena::backend::llvm