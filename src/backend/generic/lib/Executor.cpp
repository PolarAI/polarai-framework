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

#include "ExecutorImpl.hpp"

#include <polarai/backend/generic/Executor.hpp>

using namespace polarai::core;

namespace polarai::backend::generic {

void Executor::addGraph(Graph& graph) { mImpl->addGraph(graph); }

void Executor::evaluate(Graph& graph) { mImpl->evaluate(graph); }

Executor::Executor(bool enableDebugOutput, FilterFunctionT filter)
    : mImpl(polarai::utils::makeShared<ExecutorImpl>(enableDebugOutput,
                                                     std::move(filter))) {}

BackendAllocator& Executor::getAllocator() { return mImpl->getAllocator(); }
std::shared_ptr<BackendAllocator> Executor::getAllocatorPtr() {
  return mImpl->getAllocatorPtr();
}

void Executor::setAllocator(std::shared_ptr<BackendAllocator>& allocator) {
  mImpl->setAllocator(allocator);
}

std::vector<std::shared_ptr<Device>>& Executor::getDevices() {
  return mImpl->getDevices();
}
} // namespace polarai::backend::generic
