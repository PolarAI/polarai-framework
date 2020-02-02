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

#ifndef ATHENA_LEXER_H
#define ATHENA_LEXER_H

#include <llvm/ADT/StringRef.h>
#include <string>
#include <string_view>

namespace chaos {

class Token {
public:
  enum class Kind : int {
    LeftParen = '(',
    RightParen = ')',
    DoubleQuote = '"',
    SingleQuote = '\'',
    LeftSquare = '[',
    RightSquare = ']',
    LessThan = '<',
    GreaterThan = '>',
    Equal = '=',
    Plus = '+',
    Minus = '-',
    Asterisk = '*',
    Slash = '/',
    Backslash = '\\',
    Dot = '.',
    Comma = ',',
    Colon = ':',
    Semicolon = ';',
    At = '@',
    Hat = '^',
    Percent = '%',
    Hash = '#',
    EoL = '\n',

    EoF = -1,
    Unexpected = -2,

    // keywords
    And = -3,
    As = -4,
    Assert = -5,
    Break = -6,
    Class = -7,
    Continue = -6,
    Def = -7,
    Del = -8,
    Elif = -9,
    Else = -11,
    Except = -12,
    False = -13,
    Finally = -14,
    For = -15,
    From = -16,
    Global = -17,
    If = -18,
    Import = -19,
    In = -20,
    Is = -21,
    Lambda = -22,
    None = -23,
    Nonlocal = -24,
    Not = -25,
    Or = -26,
    Pass = -27,
    Raise = -28,
    Return = -29,
    True = -30,
    Try = -32,
    While = -33,
    With = -34,
    Yield = -35,

    // primary
    IntegerLiteral = -1001,
    FPLiteral = -1002,
    Identifier = -1003
  };

  explicit Token(Kind kind) : mKind(kind) {}

  Token(Kind kind, std::string lexeme)
      : mKind(kind), mLexeme(std::move(lexeme)) {}

  Token(Kind kind, const char* begin, std::size_t len)
      : mKind(kind), mLexeme(begin, len) {}

  [[nodiscard]] Kind getKind() const { return mKind; }

  [[nodiscard]] bool is(Kind kind) const { return mKind == kind; }

  [[nodiscard]] std::string_view getLexeme() { return mLexeme; }

private:
  Kind mKind;
  std::string mLexeme;
};

struct SourceLocation {
  std::shared_ptr<std::string> filename;
  std::shared_ptr<std::string> source;
  std::size_t line;
  std::size_t col;
  std::size_t lineIndentation;
};

/// The Lexer is an abstract base class providing tokenization functionality for
/// the Parser. It iterates through the stream one token at a time and keeps
/// track of lexeme location for the debugging purposes.
class Lexer {
private:
  Token mCurToken;
  std::size_t mCurLineNumber = 0;
  std::size_t mCurColNumber = 0;
  std::size_t mCurLineIndent = 0;
  int mLastChar = '\n';
  std::string mTokenPiece;

  SourceLocation mLastLocation;

  llvm::StringRef mCurLineBuffer = "\n";

  Token getToken();

  int getNextChar();

protected:
  virtual llvm::StringRef getNextLine() = 0;

public:
  explicit Lexer(std::string filename)
      : mCurToken(Token::Kind::Unexpected),
        mLastLocation{std::make_shared<std::string>(std::move(filename)),
                      std::make_shared<std::string>(""), 0, 0, 0} {}
  virtual ~Lexer() = default;

  /// \return current token in the stream.
  Token& getCurToken() { return mCurToken; }

  /// \return next token in the stream
  Token& getNextToken() { return mCurToken = getToken(); }

  void consume(const Token& token);

  SourceLocation getLastLocation() { return mLastLocation; }
};

class BufLexer : public Lexer {
private:
  const char* mCurPos;
  const char* mEnd;

protected:
  llvm::StringRef getNextLine() override;

public:
  BufLexer(const char* begin, const char* end, std::string filename)
      : Lexer(std::move(filename)), mCurPos(begin), mEnd(end) {}
};
} // namespace chaos

#endif // ATHENA_LEXER_H
