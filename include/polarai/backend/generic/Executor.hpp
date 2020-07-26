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
#include <polarai/core/graph/Graph.hpp>
#include <polarai/core/graph/Traversal.hpp>
#include <polarai/core/loader/internal/TensorAllocator.hpp>
#include <polarai/utils/Pointer.hpp>
#include <polarai/utils/storages/Vector.hpp>

namespace polarai::backend::generic {

// Forward declarations
class ExecutorImpl;
/// Default device filter. Selects all devices.
constexpr auto DefaultDeviceFilter = [](polarai::utils::SharedPtr<Device>&) {
  return true;
};

/**
 * Execute Graph with LLVM-based backend
 */
class POLAR_BACKEND_GENERIC_EXPORT Executor {
public:
  using FilterFunctionT =
      std::function<bool(polarai::utils::SharedPtr<Device>&)>;
  Executor(bool enableDebugOutput = false,
           FilterFunctionT filter = DefaultDeviceFilter);

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

  polarai::utils::Vector<std::shared_ptr<Device>>& getDevices();

private:
  polarai::utils::SharedPtr<ExecutorImpl> mImpl;
};
} // namespace polarai::backend::generic
