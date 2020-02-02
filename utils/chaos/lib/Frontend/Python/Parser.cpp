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

#include "Parser.h"
#include "AST/Decl.h"
#include "AST/Stmt.h"

namespace chaos {
chaos::FunctionDecl Parser::parseFunction() {
  auto loc = mLexer.getLastLocation();
  if (mLexer.getNextToken().getKind() != Token::Kind::Identifier) {
    llvm_unreachable("Parsing err. def not followed by decl");
  }

  std::string functionName{mLexer.getCurToken().getLexeme().data()};
  mLexer.consume(mLexer.getCurToken());

  if (mLexer.getCurToken().getKind() != Token::Kind::LeftParen) {
    llvm_unreachable("Parsing err. def id not followed by (");
  }

  FunctionDecl decl(functionName, loc);

  mLexer.consume(mLexer.getCurToken());
  while (mLexer.getCurToken().getKind() != Token::Kind::RightParen) {
    if (mLexer.getCurToken().getKind() == Token::Kind::Identifier) {
      auto param = parseParamDecl();
      decl.addParam(std::move(param));
    } else if (mLexer.getCurToken().getKind() == Token::Kind::Comma) {
      mLexer.getNextToken();
    } else {
      llvm_unreachable("Expected comma");
    }
  }

  if (mLexer.getCurToken().getKind() != Token::Kind::RightParen) {
    llvm_unreachable("Parsing err. def id not followed by )");
  }

  if (mLexer.getNextToken().getKind() != Token::Kind::Colon) {
    llvm_unreachable("Expected colon");
  }

  mLexer.getNextToken();
  auto block = parseBlock();
  decl.setBody(std::move(block));

  return std::move(decl);
}
std::unique_ptr<TranslationUnitDecl> chaos::Parser::parseTranslationUnit() {
  auto translationUnit = std::make_unique<TranslationUnitDecl>();

  auto& nextTok = mLexer.getNextToken();

  while (nextTok.getKind() != Token::Kind::EoF) {
    if (nextTok.getKind() == Token::Kind::Def) {
      auto funcDecl = parseFunction();
      DeclStmt declStmt(std::move(funcDecl));
      translationUnit->addStatement(std::move(declStmt));
    } else {
      nextTok = mLexer.getNextToken();
    }
  }

  return std::move(translationUnit);
}
AST Parser::buildAST() {
  auto translationUnit = parseTranslationUnit();
  AST ast(std::move(translationUnit));

  return std::move(ast);
}
ParamDecl Parser::parseParamDecl() {
  auto loc = mLexer.getLastLocation();
  auto curToken = mLexer.getCurToken();

  if (curToken.getKind() != Token::Kind::Identifier) {
    llvm_unreachable("Wrong param decl");
  }

  std::string paramName(curToken.getLexeme().data());
  std::string type = "__ANY_OBJECT__";

  curToken = mLexer.getNextToken();

  if (curToken.getKind() == Token::Kind::Colon) {
    curToken = mLexer.getNextToken();
    if (curToken.getKind() != Token::Kind::Identifier) {
      llvm_unreachable("Type expected");
    }
    type = curToken.getLexeme();
    mLexer.getNextToken();
  }

  return std::move(ParamDecl(paramName, type, loc));
}
BlockDecl Parser::parseBlock() {
  BlockDecl block;

  auto curIndent = mLexer.getLastLocation().lineIndentation;

  while (curIndent == mLexer.getLastLocation().lineIndentation) {
    auto statement = parseStatement();
    block.addStatement(std::move(statement));
    mLexer.getNextToken();
  }

  return std::move(block);
}
AnyStatement Parser::parseStatement() {
  if (mLexer.getCurToken().is(Token::Kind::Pass)) {
    return PassStmt();
  }

  llvm_unreachable("Unexpected statement");
}
} // namespace chaos
