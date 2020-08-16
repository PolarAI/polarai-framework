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

#include "Pattern.hpp"

#include "llvm/Support/raw_ostream.h"

namespace polarai::script {
auto operator<<(llvm::raw_ostream& os, const MatchResult& r)
    -> llvm::raw_ostream& {
  if (r) {
    os << "Success[";
  } else {
    os << "Failure[";
  }

  os << r.mPatternName << "]: " << r.mMessage;

  return os;
}

auto Pattern::eval(clang::Expr*, mlir::OpBuilder&)
    -> std::pair<MatchResult, mlir::Value> {
  return {fail("Not implemented"), mlir::Value{}};
};
} // namespace polarai::script
