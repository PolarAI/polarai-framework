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

#include "../MlirGen.hpp"

#include "TranslationUnitPattern.hpp"
#include "GPUFunctionPattern.hpp"
#include "CompoundStmtPattern.hpp"
#include "TypedefPattern.hpp"

namespace polarai::script {
void populateStandalonePatterns(PatternList& list, GenerationContext* ctx,
                                uint64_t benefit) {
  list.insertPattern<TranslationUnitPattern>(ctx, benefit);
  list.insertPattern<GPUFunctionPattern>(ctx, benefit);
  list.insertPattern<TypedefPattern>(ctx, benefit);
  list.insertPattern<CompoundStmtPattern>(ctx, benefit);
}
} // namespace polarai::script
