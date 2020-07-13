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
namespace polarai::core {
class AbstractNode;
class Node;
class InputNode;
class OutputNodeInternal;
class LossNode;
namespace impl {
class GraphImpl;
class ContextImpl;

class AbstractNodeImpl;
class NodeImpl;
class OutputNodeImpl;
} // namespace impl
namespace internal {
class GraphInternal;
class ContextInternal;

class AbstractNodeInternal;
class NodeInternal;
class InputNodeInternal;
class OutputNodeInternal;
} // namespace internal
} // namespace polarai::core
