/*
 * Copyright (c) 2018 Athena. All rights reserved.
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

#include <athena/core/operation/internal/OperationInternal.h>

namespace athena::core::internal {
OperationInternal::OperationInternal(utils::WeakPtr<ContextInternal> context,
                                     utils::Index publicNodeIndex,
                                     utils::String name)
    : Entity(std::move(context), publicNodeIndex, std::move(name)) {}

void OperationInternal::lockTensors(core::internal::Generator& generator, std::unordered_map<utils::Index, GenValue> argMap,
                                    std::unordered_map<utils::Index, GenValue> resultMap) const {
  generator.callBuiltin<builtin::Lock>(resultMap.begin()->second, LockType::READ_WRITE);
  argMap.erase(resultMap.begin()->first);
  for (auto& [_, genValue] : argMap) {
    generator.callBuiltin<builtin::Lock>(genValue, LockType::READ);
  }
}

void OperationInternal::releaseTensors(core::internal::Generator& generator, std::unordered_map<utils::Index, GenValue> argMap,
                                       std::unordered_map<utils::Index, GenValue> resultMap) const {
  argMap.erase(resultMap.begin()->first);
  for (auto& [_, genValue] : argMap) {
    generator.callBuiltin<builtin::Release>(genValue);
  }
  for (auto& [_, genValue] : resultMap) {
    generator.callBuiltin<builtin::Release>(genValue);
  }
}
} // namespace athena::core::internal
