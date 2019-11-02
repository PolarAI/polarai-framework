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

#include <athena/core/Allocator.h>
#include <athena/core/Executor.h>
#include <athena/core/Traversal.h>

namespace llvm {
class Module;
}

namespace athena::backend::llvm {

// Forward declarations
class AthenaJIT;
class RuntimeDriver;

/**
 * Execute Graph with LLVM-based backend
 */
class LLVMExecutor : public athena::core::Executor {
    private:
    std::shared_ptr<AthenaJIT> mJITCompiler{nullptr};
    std::unique_ptr<core::Allocator> mAllocator;
    std::shared_ptr<RuntimeDriver> mRuntimeDriver;

    template <typename T>
    using ClusterContainer = std::vector<core::inner::NodeDependencies<T>>;

    /**
     * Generate LLVM IR for Graph
     * @param graph Execution graph
     * @return LLVM IR module with generated graph
     */
    std::vector<std::unique_ptr<::llvm::Module>> compileGraph(
        athena::core::Graph &graph);

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
    void evaluate() override;

    /**
     * Apply weight correction technique
     */
    void optimizeGraph() override;

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
