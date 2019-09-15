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
#ifndef ATHENA_LLVMEXECUTOR_H
#define ATHENA_LLVMEXECUTOR_H

#include <athena/backend/llvm/AthenaJIT.h>
#include <athena/backend/llvm/LLVMGenerator.h>
#include <athena/backend/llvm/runtime-driver/runtime-driver.h>
#include <athena/core/Allocator.h>
#include <athena/core/Executor.h>
#include <athena/core/Traversal.h>

#include <llvm/IR/LLVMContext.h>
#include <llvm/IR/Module.h>

namespace athena::backend::llvm {

/**
 * Execute Graph with LLVM-based backend
 */
class LLVMExecutor : public athena::core::Executor {
    private:
    std::unique_ptr<AthenaJIT> mJITCompiler;
    //    std::unique_ptr<::llvm::Module> mMainModule;
    std::unique_ptr<core::Allocator> mAllocator;
    //    athena::core::Traversal mGraphTraversal;
    std::unique_ptr<RuntimeDriver> mRuntimeDriver;

    std::vector<std::unique_ptr<::llvm::Module>> mExistingModules;

    template <typename T>
    using ClusterContainer = std::vector<core::inner::NodeDependencies<T>>;

    /**
     * Generate LLVM IR for Graph
     * @param graph Execution graph
     * @return LLVM IR module with generated graph
     */
    std::vector<std::unique_ptr<::llvm::Module>> compileGraph(
        athena::core::Graph &graph);

    /**
     * Generate LLVM IR for input nodes
     * @param generator Generator associated with corresponding graph
     * @param inputNodes InputNodes that need to be compiled
     */
    static void compileInputNodes(
        LLVMGenerator &generator,
        const ClusterContainer<core::InputNode> &inputNodes);

    /**
     * Generate LLVM IR for nodes
     * @param generator Generator associated with corresponding graph
     * @param inputNodes Nodes that need to be compiled
     */
    static void compileActionNodes(
        LLVMGenerator &generator,
        const ClusterContainer<core::Node> &actionNodes);

    /**
     * Generate LLVM IR for loss nodes
     * @param generator Generator associated with corresponding graph
     * @param inputNodes LossNodes that need to be compiled
     */
    static void compileLossNodes(
        LLVMGenerator &generator,
        const ClusterContainer<core::LossNode> &lossNodes);

    static void compileDerivatives(LLVMGenerator &generator,
                                   const core::Traversal &traversal,
                            core::Optimizer &graphOptimizer);

    static void compileLossDerivatives(
        LLVMGenerator &generator,
        const ClusterContainer<core::LossNode> &lossNodes,
        core::Optimizer &graphOptimizer);

    static void compileNodeDerivatives(
        LLVMGenerator &generator,
        const ClusterContainer<core::Node> &nodes,
        core::Optimizer &graphOptimizer);

    static void adjustWeights(
        LLVMGenerator &generator,
        const ClusterContainer<core::InputNode> &inputNodes,
        core::Optimizer &graphOptimizer);

    public:
    LLVMExecutor();
    /**
     * Set Graph for execution. This method must always be called before
     * execution
     * @param graph Graph to be executed
     */
    void setGraph(athena::core::Graph &graph) override;
    /**
     * Do actual computation
     */
    void execute() override;
    /**
     *
     * @return Associated Allocator
     */
    std::unique_ptr<core::Allocator> &getAllocator();
    /**
     * Set Allocator to be used
     * @param allocator User Allocator
     */
    void setAllocator(std::unique_ptr<core::Allocator> &allocator);
};
}  // namespace athena::backend::llvm

#endif  // ATHENA_LLVMEXECUTOR_H
