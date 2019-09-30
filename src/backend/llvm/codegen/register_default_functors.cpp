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

#include "register_default_functors.h"

#include "common.h"

namespace athena::backend::llvm::codegen {
void registerDefaultFunctors(LLVMGenerator *generator) {
    registerStandardBuiltin<BuiltinThreeTensorArgs>("add", generator);
    registerAllocate(generator);
    registerFill(generator);
    registerStandardBuiltin<BuiltinTATATArgs>("hadamard", generator);
    registerStandardBuiltin<BuiltinTATATArgs>("fma", generator);
    registerStandardBuiltin<BuiltinThreeTensorArgs>("mse", generator);
}
}  // namespace athena::backend::llvm::codegen