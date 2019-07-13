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
#include <athena/backend/llvm/runtime-driver/runtime-driver.h>
#include <athena/core/Allocator.h>
#include <athena/core/Executor.h>

#include <llvm/IR/LLVMContext.h>
#include <llvm/IR/Module.h>

namespace athena::backend::llvm {

/**
 * Execute Graph with LLVM-based backend
 */
class LLVMExecutor : public athena::core::Executor {
    private:
    std::unique_ptr<AthenaJIT> mJITCompiler;
    std::unique_ptr<::llvm::Module> mMainModule;
    std::unique_ptr<core::Allocator> mAllocator;
    athena::core::Traversal mGraphTraversal;
    std::unique_ptr<RuntimeDriver> mRuntimeDriver;

    std::vector<std::unique_ptr<::llvm::Module>> mExistingModules;

    public:
    LLVMExecutor();
    /**
     * Prepare Graph for execution. This method must always be called before
     * execution
     * @param graph Graph to be executed
     */
    void prepare(athena::core::Graph &graph) override;
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
