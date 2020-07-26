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

#pragma once

#include <polar_backend_generic_export.h>
#include <polarai/backend/generic/BackendAllocator.hpp>
#include <polarai/backend/generic/runtime/Device.hpp>
#include <polarai/core/graph/Traversal.hpp>
#include <polarai/core/internal/Executor.hpp>
#include <polarai/core/loader/internal/TensorAllocator.hpp>
#include <polarai/utils/Pointer.hpp>
#include <polarai/utils/storages/Vector.hpp>

namespace polarai::backend::generic {

// Forward declarations
class PolarJIT;
class RuntimeDriver;

/**
 * Execute Graph with LLVM-based backend
 */
class POLAR_BACKEND_GENERIC_EXPORT ExecutorImpl {
public:
  using FilterFunctionT =
      std::function<bool(polarai::utils::SharedPtr<Device>&)>;
  ExecutorImpl(bool enableDebugOutput, FilterFunctionT filter);

  /// Adds Graph to compilable module.
  ///
  /// \param graph is a valid Graph to be compiled.
  void addGraph(polarai::core::Graph& graph);

  /// Executes particular graph.
  ///
  /// \param graph is a valid Graph, that has been previously added.
  void evaluate(polarai::core::Graph& graph);

  BackendAllocator& getAllocator();
  polarai::utils::SharedPtr<BackendAllocator> getAllocatorPtr();
  void setAllocator(polarai::utils::SharedPtr<BackendAllocator>& allocator);

  polarai::utils::Vector<polarai::utils::SharedPtr<Device>>& getDevices();

private:
  FilterFunctionT mFilter;
  // Driver must come first to ensure proper shutdown
  std::shared_ptr<RuntimeDriver> mRuntimeDriver;
  std::vector<std::shared_ptr<Device>> mDevices;

  std::shared_ptr<PolarJIT> mJITCompiler{nullptr};
  std::shared_ptr<BackendAllocator> mAllocator;
};
} // namespace polarai::backend::generic
