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

#include "Decl.h"
#include "Stmt.h"

namespace chaos {
void TranslationUnitDecl::dump(std::size_t tab, bool isFirst) const {
  llvm::outs() << std::string(tab - isFirst, ' ') << (isFirst ? "`" : "")
               << "|TranslationUnitDecl\n";
  bool newFirst = true;
  for (auto& statement : mStatements) {
    impl::dump(statement, tab + 1, newFirst);
    newFirst = false;
  }
}
void FunctionDecl::dump(std::size_t tab, bool isFirst) const {
  llvm::outs() << std::string(tab - isFirst, ' ') << (isFirst ? "`" : "")
               << "|FunctionDecl: " << mFunctionName << "\n";

  bool newFirst = true;
  for (auto& param : mParams) {
    param.dump(tab + 1, newFirst);
    newFirst = false;
  }
  mBody.dump(tab + 1);
}
void FunctionDecl::addParam(ParamDecl paramDecl) {
  mParams.push_back(std::move(paramDecl));
}
void ParamDecl::dump(std::size_t tab, bool isFirst) const {
  llvm::outs() << std::string(tab - isFirst, ' ') << (isFirst ? "`" : "")
               << "|ParamDecl: " << mParamName << "<" << mType << ">\n";
}
void BlockDecl::dump(std::size_t tab, bool isFirst) const {
  llvm::outs() << std::string(tab - isFirst, ' ') << (isFirst ? "`" : "")
               << "|BlockDecl\n";
  bool newFirst = true;
  for (auto& stmt : mStatements) {
    impl::dump(stmt, tab + 1, newFirst);
    newFirst = false;
  }
}
void BlockDecl::addStatement(AnyStatement stmt) {
  mStatements.push_back(std::move(stmt));
}
void FunctionDecl::setBody(BlockDecl blockDecl) {
  mBody = std::move(blockDecl);
}
} // namespace chaos