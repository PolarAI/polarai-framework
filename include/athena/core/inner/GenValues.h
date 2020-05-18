//===----------------------------------------------------------------------===//
// Copyright (c) 2020 Athena. All rights reserved.
// https://getathena.ml
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

#include <memory>

namespace athena::core {
class GenValue;
namespace inner {
class GenValueImplBase {};
class GenGraphImplBase {};
class GenInsPointImplBase {};
class GenNodeImplBase {
public:
  virtual auto getOperand(size_t) -> GenValue = 0;
  virtual auto getResult() -> GenValue = 0;
  virtual auto getBatchIndex() -> GenValue = 0;
};
} // namespace inner
class GenValue {
  std::shared_ptr<inner::GenValueImplBase> mValue;

public:
  GenValue(std::shared_ptr<inner::GenValueImplBase> val)
      : mValue(std::move(val)) {}
  template <typename T> auto value() const -> T& {
    return *static_cast<T*>(mValue.get());
  }
};

class GenNode {
  std::shared_ptr<inner::GenNodeImplBase> mNode;

public:
  GenNode(std::shared_ptr<inner::GenNodeImplBase> node)
      : mNode(std::move(node)) {}
  template <typename T> auto node() const -> T& {
    return *static_cast<T*>(mNode.get());
  }
  auto getOperand(size_t i) -> GenValue { return mNode->getOperand(i); }
  auto getResult() -> GenValue { return mNode->getResult(); }
  auto getBatchIndex() -> GenValue { return mNode->getBatchIndex(); }
};

class GenInsertionPoint {
  std::shared_ptr<inner::GenInsPointImplBase> mPoint;

public:
  GenInsertionPoint(std::shared_ptr<inner::GenInsPointImplBase> point)
      : mPoint(std::move(point)) {}
  template <typename T> auto point() const -> T& {
    return *static_cast<T*>(mPoint.get());
  }
};

class GenGraph {
  std::shared_ptr<inner::GenGraphImplBase> mGraph;

public:
  GenGraph(std::shared_ptr<inner::GenGraphImplBase> graph)
      : mGraph(std::move(graph)) {}
  template <typename T> auto graph() -> T& {
    return *static_cast<T*>(mGraph.get());
  }
};
} // namespace athena::core