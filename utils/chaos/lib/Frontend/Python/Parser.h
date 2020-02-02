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

#ifndef ATHENA_PARSER_H
#define ATHENA_PARSER_H

#include "AST/AST.h"
#include "AST/Decl.h"
#include "Lexer.h"

namespace chaos {
/// This class implements a grammar parser for Python 3.8.1 The full grammar
/// description is taken from https://devguide.python.org/grammar/.
class Parser {
private:
  Lexer& mLexer;

  /// Parses function declaration
  /// funcdef: 'def' NAME parameters ':' func_body
  FunctionDecl parseFunction();

  /// Parses function parameters declaration
  /// param: identifier [':' TYPE]
  ParamDecl parseParamDecl();

  BlockDecl parseBlock();

  AnyStatement parseStatement();

  std::unique_ptr<TranslationUnitDecl> parseTranslationUnit();

public:
  explicit Parser(Lexer& lexer) : mLexer(lexer) {}

  AST buildAST();
};
} // namespace chaos

#endif // ATHENA_PARSER_H
