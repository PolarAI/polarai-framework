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

#include <athena/backend/llvm/BackendAllocator.h>
#include <athena/backend/llvm/llvm_export.h>
#include <athena/core/graph/Traversal.h>
#include <athena/core/internal/Executor.h>
#include <athena/core/loader/internal/TensorAllocator.h>

namespace llvm {
class Module;
}

namespace athena::backend::llvm {

// Forward declarations
class AthenaJIT;
class LegacyRuntimeDriver;

/**
 * Execute Graph with LLVM-based backend
 */
class ATH_BACKEND_LLVM_EXPORT LLVMExecutor
    : public athena::core::internal::Executor {
private:
  std::shared_ptr<AthenaJIT> mJITCompiler{nullptr};
  std::unique_ptr<BackendAllocator> mAllocator;
  std::shared_ptr<LegacyRuntimeDriver> mRuntimeDriver;

  /**
   * Generate LLVM IR for Graph
   * @param graph Execution graph
   * @return LLVM IR module with generated graph
   */
  std::vector<std::unique_ptr<::llvm::Module>>
  compileGraph(athena::core::Graph& graph);

public:
  LLVMExecutor();
  /**
   * Set Graph for execution. This method must always be called before
   * execution
   * @param graph Graph to be executed
   */
  void setGraph(athena::core::Graph& graph) override;
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
  BackendAllocator& getAllocator();
  /**
   * Set Allocator to be used
   * @param allocator User Allocator
   */
  void setAllocator(std::unique_ptr<BackendAllocator>& allocator);
};
} // namespace athena::backend::llvm

#endif // ATHENA_LLVMEXECUTOR_H
