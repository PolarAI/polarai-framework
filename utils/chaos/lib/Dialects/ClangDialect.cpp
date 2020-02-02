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

#include <Dialects/ClangDialect.h>

clang::ClangDialect::ClangDialect(mlir::MLIRContext* ctx)
    : mlir::Dialect("clang", ctx) {
  addOperations<
#define GET_OP_LIST
#include <Dialects/ClangDialect.cpp.inc>
      >();
}

namespace clang {
using namespace llvm;
#define GET_OP_CLASSES
#include <Dialects/ClangDialect.cpp.inc>
} // namespace clang

void clang::ForOp::build(Builder* builder, OperationState& result) {
  auto init = result.addRegion();
  init->push_back(new Block);
  auto cond = result.addRegion();
  cond->push_back(new Block);
  auto inc = result.addRegion();
  inc->push_back(new Block);
  auto body = result.addRegion();
  body->push_back(new Block);
}