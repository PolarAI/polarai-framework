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

#include "Lexer.h"

#include <llvm/ADT/StringSwitch.h>

namespace chaos {
int Lexer::getNextChar() {
  if (mCurLineBuffer.empty()) {
    return static_cast<int>(Token::Kind::EoF);
  }
  mCurColNumber++;

  auto nextChar = mCurLineBuffer.front();
  mCurLineBuffer = mCurLineBuffer.drop_front();

  if (mCurLineBuffer.empty()) {
    mCurColNumber = 0;
    mCurLineIndent = 0;
    mCurLineBuffer = getNextLine();
  }

  if (nextChar == '\n') {
    mCurLineNumber++;
    mCurColNumber = 0;
  }

  return nextChar;
}

static Token::Kind stringToTokenKind(std::string_view str) {
  using K = Token::Kind;

  K finalKind = llvm::StringSwitch<K>(str.data())
                    .Case("and", K::And)
                    .Case("as", K::As)
                    .Case("assert", K::Assert)
                    .Case("break", K::Break)
                    .Case("class", K::Class)
                    .Case("continue", K::Continue)
                    .Case("def", K::Def)
                    .Case("del", K::Del)
                    .Case("elif", K::Elif)
                    .Case("else", K::Else)
                    .Case("except", K::Except)
                    .Case("False", K::False)
                    .Case("finally", K::Finally)
                    .Case("for", K::For)
                    .Case("from", K::From)
                    .Case("global", K::Global)
                    .Case("if", K::If)
                    .Case("import", K::Import)
                    .Case("in", K::In)
                    .Case("is", K::Is)
                    .Case("lambda", K::Lambda)
                    .Case("None", K::None)
                    .Case("nonlocal", K::Nonlocal)
                    .Case("Not", K::Not)
                    .Case("or", K::Or)
                    .Case("pass", K::Pass)
                    .Case("raise", K::Raise)
                    .Case("True", K::True)
                    .Case("try", K::Try)
                    .Case("while", K::While)
                    .Case("with", K::With)
                    .Case("yield", K::Yield)
                    .Default(K::Unexpected);

  return finalKind;
}

Token Lexer::getToken() {
  bool shouldCountIndentation = mCurColNumber == 0;

  while (isspace(mLastChar)) {
    if (shouldCountIndentation)
      mCurLineIndent++;
    mLastChar = getNextChar();
  }

  mLastLocation.line = mCurLineNumber;
  mLastLocation.col = mCurColNumber;
  mLastLocation.lineIndentation = mCurLineIndent;

  if (isalpha(mLastChar)) {
    mTokenPiece = (char)mLastChar;
    while (isalnum(mLastChar = getNextChar()) || mLastChar == '_') {
      mTokenPiece += (char)mLastChar;
    }

    auto kind = stringToTokenKind(mTokenPiece);
    if (kind != Token::Kind::Unexpected) {
      mTokenPiece = "";
      return Token(kind);
    }
    Token res(Token::Kind::Identifier, std::move(mTokenPiece));
    mTokenPiece = "";

    return res;
  }

  bool containsDot = false;
  // fixme negative numbers
  if (isdigit(mLastChar) || mLastChar == '.') {
    if (mLastChar == '.')
      containsDot = true;

    mTokenPiece += (char)mLastChar;

    while (isdigit(mLastChar) || mLastChar == '.') {
      mTokenPiece += (char)mLastChar;
    }

    Token res(containsDot ? Token::Kind::FPLiteral
                          : Token::Kind::IntegerLiteral,
              mTokenPiece);
    mTokenPiece = "";

    return res;
  }

  if (mLastChar == '#') {
    do {
      mLastChar = getNextChar();
    } while (mLastChar != -1 && mLastChar != '\n' && mLastChar != 'r');

    if (mLastChar != -1) {
      return getToken();
    }
  }

  if (mLastChar == '(') {
    mLastChar = getNextChar();
    return Token(Token::Kind::LeftParen);
  }

  if (mLastChar == ')') {
    mLastChar = getNextChar();
    return Token(Token::Kind::RightParen);
  }

  if (mLastChar == ',') {
    mLastChar = getNextChar();
    return Token(Token::Kind::Comma);
  }

  if (mLastChar == ':') {
    mLastChar = getNextChar();
    return Token(Token::Kind::Colon);
  }

  if (mLastChar == -1) {
    return Token(Token::Kind::EoF);
  }

  if (mLastChar == '\n') {
    mLastChar = getNextChar();
    return Token(Token::Kind::EoL);
  }

  // fixme support for:
  //  1. String literals;
  //  2. Packages.

  return Token(Token::Kind::Unexpected);
}
void Lexer::consume(const Token& token) { getNextToken(); }
llvm::StringRef BufLexer::getNextLine() {
  auto* begin = mCurPos;
  while (mCurPos <= mEnd && *mCurPos && *mCurPos != '\n') {
    ++mCurPos;
  }

  if (mCurPos <= mEnd && *mCurPos) {
    ++mCurPos;
  }

  llvm::StringRef result{begin, static_cast<size_t>(mCurPos - begin)};
  return result;
}
} // namespace chaos
