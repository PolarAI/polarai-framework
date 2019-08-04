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
#include <athena/core/LossNode.h>
#include <athena/core/Node.h>
#include <athena/core/Optimizer.h>
#include <athena/core/inner/GlobalTables.h>
#include <athena/core/inner/InnerFunctions.h>

#include "llvm/ExecutionEngine/ExecutionEngine.h"
#include "llvm/Support/TargetSelect.h"

#include <cassert>

namespace athena::backend::llvm {

void LLVMExecutor::setGraph(athena::core::Graph &graph) {
    auto modules = compileGraph(graph);

    // At the moment compileGraph method always returns exactly 1 module.
    // That may change in future when we decide to go with a more complex
    // structure of neural networks.
    for (auto &module : modules) {
        auto err = mJITCompiler->addModule(module);
        if (err) {
            core::FatalError(1, "Error adding module to JIT");
        }
    }

    // Prepare runtime library
    for (auto &module : mRuntimeDriver->getModules()) {
        auto err = mJITCompiler->addModule(module);
        if (err) {
            new core::FatalError(1, "Unable to add module");
        }
    }
}

void LLVMExecutor::execute() {
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

std::vector<std::unique_ptr<::llvm::Module>> LLVMExecutor::compileGraph(
    athena::core::Graph &graph) {
    auto llvmModule = std::make_unique<::llvm::Module>(
        graph.getGraphName(), mJITCompiler->getContext());

    llvmModule->setDataLayout(mJITCompiler->getDataLayout());
    // TODO get real target triple
    llvmModule->setTargetTriple(::llvm::sys::getDefaultTargetTriple());

    ::llvm::FunctionType *FT = ::llvm::FunctionType::get(
        ::llvm::Type::getVoidTy(mJITCompiler->getContext()), false);
    ::llvm::Function::Create(FT, ::llvm::Function::ExternalLinkage, "jitmain",
                             *llvmModule);

    LLVMGenerator generator(mJITCompiler->getContext(), llvmModule, *mAllocator,
                            mRuntimeDriver->getModules());

    auto graphTraversal = graph.traverse();

    for (auto &cluster : graphTraversal.getClusters()) {
        auto &inputNodes = cluster.get<core::InputNode>();
        compileInputNodes(generator, inputNodes);

        auto &actionNodes = cluster.get<core::Node>();
        compileActionNodes(generator, actionNodes);

        auto &lossNodes = cluster.get<core::LossNode>();
        compileLossNodes(generator, lossNodes);
    }

    compileDerivatives(generator, graphTraversal, *graph.getOptimizer());

    auto builder = generator.getBuilder();

    builder.CreateRetVoid();

    std::vector<std::unique_ptr<::llvm::Module>> resultModules;
    resultModules.push_back(std::move(llvmModule));

    return resultModules;
}
void LLVMExecutor::compileInputNodes(
    LLVMGenerator &generator,
    const LLVMExecutor::ClusterContainer<core::InputNode> &inputNodes) {
    for (auto &nodeDeps : inputNodes) {
        auto &inputNode = node_cast<core::InputNode &>(
            *core::inner::getNodeTable()[nodeDeps.nodeIndex]);
        generator.openNode(inputNode.getName());
        generator.generate("allocate",
                           core::inner::getTensorFromNode(inputNode));
        generator.generateLoad(inputNode.getLoader(),
                               core::inner::getTensorFromNode(inputNode));
        generator.closeNode();
    }
}
void LLVMExecutor::compileActionNodes(
    LLVMGenerator &generator,
    const LLVMExecutor::ClusterContainer<core::Node> &actionNodes) {
    for (auto &nodeDeps : actionNodes) {
        std::vector<core::inner::Tensor *> preparedTensors;
        for (auto &input : nodeDeps.input) {
            auto *node = core::inner::getNodeTable()[input.nodeIndex];
            preparedTensors.push_back(&core::inner::getTensorFromNode(*node));
        }
        auto &node = node_cast<core::Node &>(
            *core::inner::getNodeTable()[nodeDeps.nodeIndex]);
        generator.openNode(node.getName());
        generator.generate("allocate", core::inner::getTensorFromNode(node));
        preparedTensors.push_back(&core::inner::getTensorFromNode(node));
        // todo lock tensors in memory
        node.getOperation().gen(generator, preparedTensors);
        // todo unlock tensors in memory

        generator.closeNode();
    }
}
void LLVMExecutor::compileLossNodes(
    LLVMGenerator &generator,
    const LLVMExecutor::ClusterContainer<core::LossNode> &lossNodes) {
    if (lossNodes.size() == 1) {
        auto &nodeDeps = lossNodes[0];
        std::vector<core::inner::Tensor *> preparedTensors;
        for (auto &input : nodeDeps.input) {
            auto *node = core::inner::getNodeTable()[input.nodeIndex];
            preparedTensors.push_back(&core::inner::getTensorFromNode(*node));
        }
        auto &node = *reinterpret_cast<core::LossNode *>(
            core::inner::getNodeTable()[nodeDeps.nodeIndex]);
        generator.openNode(node.getName());
        generator.generate("allocate", core::inner::getTensorFromNode(node));
        preparedTensors.push_back(&core::inner::getTensorFromNode(node));
        // todo lock tensors in memory
        node.getOperation().gen(generator, preparedTensors);
        // todo unlock tensors in memory

        generator.closeNode();
    } else if (lossNodes.size() > 1) {
        new core::FatalError(1, "More than 1 loss node");
    }
}
void LLVMExecutor::compileDerivatives(LLVMGenerator &generator,
                                      const core::Traversal &traversal,
                                      core::Optimizer &graphOptimizer) {
    auto clusters = traversal.getClusters();

    std::unordered_map<core::AbstractNode *, std::vector<core::inner::Tensor *>>
        nodeErrors;

    for (auto clusterIt = clusters.rbegin(); clusterIt != clusters.rend();
         ++clusterIt) {
        // Generate dericatives for loss nodes. There is no need to calculate
        // error, as loss node output is error.
        auto &lossNodes = clusterIt->get<core::LossNode>();
        for (auto &nodeDeps : lossNodes) {
            // Collect inputs
            std::vector<core::inner::Tensor *> inputs;
            for (auto &inp : nodeDeps.input) {
                auto &tensor = core::inner::getTensorFromNode(
                    *core::inner::getNodeTable()[inp.nodeIndex]);

                inputs.push_back(&tensor);
            }

            auto &lossNode = node_cast<core::LossNode &>(
                *core::inner::getNodeTable()[nodeDeps.nodeIndex]);

            auto &outputTensor = core::inner::getTensorFromNode(lossNode);

            // Calculate derivatives with respect to inputs
            // TODO consider moving this computation to forward feed since it
            // seems to have all the necessary info.
            for (size_t idx = 0;
                 idx < lossNode.getOperation().getOperandsCount(); idx++) {
                auto derivativeTensor =
                    core::inner::getDerivativeTensor(lossNode, idx);

                generator.generate("allocate", derivativeTensor);
                // todo lock tensors in memory
                lossNode.getOperation().genDerivative(
                    graphOptimizer.getRequiredOrder(), generator, outputTensor,
                    inputs, derivativeTensor, idx);

                // TODO memory clean up

                // As the final derivative will be just sum of partial
                // derivatives, we don't really care about order here
                auto *errNode =
                    core::inner::getNodeTable()[nodeDeps.input[idx].nodeIndex];
                nodeErrors.insert(std::make_pair(
                    errNode, std::vector<core::inner::Tensor *>()));
                nodeErrors[errNode].push_back(
                    &core::inner::getErrorTensor(lossNode, idx));
            }
        }

        // Generate dericatives and errors for action nodes.
        auto &actionNodes = clusterIt->get<core::Node>();
        for (auto &nodeDeps : actionNodes) {
            // Collect inputs
            std::vector<core::inner::Tensor *> inputs;
            for (auto &inp : nodeDeps.input) {
                auto &tensor = core::inner::getTensorFromNode(
                    *core::inner::getNodeTable()[inp.nodeIndex]);

                inputs.push_back(&tensor);
            }

            auto &node = node_cast<core::Node &>(
                *core::inner::getNodeTable()[nodeDeps.nodeIndex]);

            auto &outputTensor = core::inner::getTensorFromNode(node);
            // TODO this looks dumb to me. Maybe just use node's vectors?
            std::vector<core::inner::Tensor *> derivativeTensors;
            std::vector<core::inner::Tensor *> errorTensors;

            for (size_t idx = 0; idx < node.getOperation().getOperandsCount();
                 idx++) {
                auto &derivativeTensor =
                    core::inner::getDerivativeTensor(node, idx);

                generator.generate("allocate", derivativeTensor);
                // todo lock tensors in memory
                node.getOperation().genDerivative(
                    graphOptimizer.getRequiredOrder(), generator, outputTensor,
                    inputs, derivativeTensor, idx);

                // As the final derivative will be just sum of partial
                // derivatives, we don't really care about order here
                auto *errNode =
                    core::inner::getNodeTable()[nodeDeps.input[idx].nodeIndex];
                nodeErrors.insert(std::make_pair(
                    errNode, std::vector<core::inner::Tensor *>()));
                nodeErrors[errNode].push_back(
                    &core::inner::getErrorTensor(node, idx));

                derivativeTensors.push_back(
                    &core::inner::getDerivativeTensor(node, idx));
                errorTensors.push_back(&core::inner::getErrorTensor(node, idx));
            }

            for (auto *errorTensor : nodeErrors[&node]) {
                generator.generate("allocate", *errorTensor);
                // todo lock in memory
            }

            // Actually generate errors
            graphOptimizer.genErrors(generator, derivativeTensors, errorTensors,
                                     nodeErrors[&node]);
            // TODO cleanup wasted tensors
        }

        auto &inputNodes = clusterIt->get<core::InputNode>();
        for (auto &nodeDeps : inputNodes) {
            auto &inputNode = node_cast<core::InputNode &>(
                *core::inner::getNodeTable()[nodeDeps.nodeIndex]);

            // Frozen nodes are usually user data thus not updated
            if (inputNode.isFrozen()) continue;

            // todo lock in memory
            auto &tensor = core::inner::getTensorFromNode(inputNode);

            // Apply error correction
            graphOptimizer.genFix(generator, tensor, nodeErrors[&inputNode]);
        }
    }
}

}  // namespace athena::backend::llvm