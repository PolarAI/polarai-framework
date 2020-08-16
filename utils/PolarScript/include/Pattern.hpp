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

#pragma once

#include <cstddef>
#include <iostream>
#include <utility>

#include "llvm/ADT/StringRef.h"
#include "mlir/IR/Module.h"

namespace clang {
class Decl;
class Stmt;
class Expr;
} // namespace clang

namespace mlir {
class OpBuilder;
class Value;
} // namespace mlir

namespace polarai::script {
class GenerationContext;

class MatchResult {
public:
  constexpr operator bool() const { return mSuccess; }

private:
  friend class Pattern;
  friend llvm::raw_ostream& operator<<(llvm::raw_ostream&, const MatchResult&);

  MatchResult(bool success, llvm::StringRef patternName,
              llvm::StringRef message)
      : mSuccess(success), mPatternName(patternName), mMessage(message) {}

  bool mSuccess;
  llvm::StringRef mPatternName;
  llvm::StringRef mMessage;
};

llvm::raw_ostream& operator<<(llvm::raw_ostream&, const MatchResult&);

class Pattern {
public:
  Pattern(GenerationContext* ctx, uint64_t benefit)
      : mContext(ctx), mBenefit(benefit) {}

  virtual llvm::StringRef getPatternName() { return "Unknown"; }

  virtual MatchResult generate(clang::Decl*, mlir::OpBuilder&) {
    return fail("Not implemented");
  };
  virtual MatchResult generate(clang::Stmt*, mlir::OpBuilder&) {
    return fail("Not implemented");
  };
  virtual std::pair<MatchResult, mlir::Value> eval(clang::Expr*,
                                                   mlir::OpBuilder&);

  uint64_t getBenefit() const { return mBenefit; }

  GenerationContext* getContext() const { return mContext; }

protected:
  MatchResult fail(llvm::StringRef message) {
    return MatchResult(false, getPatternName(), message);
  }
  MatchResult success() { return MatchResult(true, getPatternName(), ""); }

private:
  GenerationContext* mContext;
  uint64_t mBenefit;
};

template <typename T> class TypedPattern : public Pattern {
public:
  TypedPattern(GenerationContext* ctx, uint64_t benefit)
      : Pattern(ctx, benefit) {}
  virtual MatchResult generate(T*, mlir::OpBuilder&) {
    return fail("Not implemented");
  };

  MatchResult generate(clang::Decl* decl, mlir::OpBuilder& builder) override {
    if constexpr (std::is_base_of_v<clang::Decl, T>) {
      if (llvm::isa<T>(decl)) {
        return generate(llvm::cast<T>(decl), builder);
      }
    }
    return fail("Type mismatch");
  }

  MatchResult generate(clang::Stmt* stmt, mlir::OpBuilder& builder) override {
    if constexpr (std::is_base_of_v<clang::Stmt, T>) {
      if (llvm::isa<T>(stmt)) {
        return generate(llvm::cast<T>(stmt), builder);
      }
    }
    return fail("Type mismatch");
  }

  std::pair<MatchResult, mlir::Value> eval(clang::Expr* expr,
                                    mlir::OpBuilder& builder) override {
    if constexpr (std::is_base_of_v<clang::Expr*, T>) {
      if (llvm::isa<T>(expr)) {
        return eval(llvm::cast<T>(expr), builder);
      }
    }
    return {fail("Type mismatch"), {}};
  };

  virtual std::pair<MatchResult, mlir::Value> eval(T*, mlir::OpBuilder&) {
    return {fail("Not implemented"), {}};
  };
};

class PatternList {
public:
  PatternList() = default;
  PatternList(const PatternList&) = delete;
  PatternList(PatternList&&) = default;

  void generate(clang::Decl* decl, mlir::OpBuilder& builder) {
    std::cerr << "[gendecl] mPatterns.size() = " << mPatterns.size();
    int i = 0;
    for (auto& pattern : mPatterns) {
      std::clog << ++i << '\n';
      MatchResult res = pattern->generate(decl, builder);
      if (res) {
        return;
      } else {
        llvm::errs() << res << '\n';
      }
    }

    // decl->dump();
    llvm::errs() << "Tried " << i << " patterns\n\n";
    llvm_unreachable("No matching pattern for declaration");
  }
  void generate(clang::Stmt* decl, mlir::OpBuilder& builder) {
    std::cerr << "[genstmt] mPatterns.size() = " << mPatterns.size();
    for (auto& pattern : mPatterns) {
      if (pattern->generate(decl, builder)) {
        return;
      }
    }

    llvm_unreachable("No matching pattern for satement");
  }
  mlir::Value eval(clang::Expr* expr, mlir::OpBuilder& builder) {
    std::cerr << "[eval] mPatterns.size() = " << mPatterns.size();
    for (auto& pattern : mPatterns) {
      auto res = pattern->eval(expr, builder);
      if (res.first) {
        return res.second;
      }
    }
    llvm_unreachable("No matching pattern for expression");
  };

  template <typename T, typename... Args> void insertPattern(Args&&... args) {
    mPatterns.push_back(std::make_unique<T>(std::forward<Args>(args)...));
    std::sort(mPatterns.begin(), mPatterns.end(),
              [](std::unique_ptr<Pattern>& a, std::unique_ptr<Pattern>& b) {
                return a->getBenefit() > b->getBenefit();
              });
    std::cerr << "mPatterns.size() = " << mPatterns.size();
  };

private:
  std::vector<std::unique_ptr<Pattern>> mPatterns;
};
} // namespace polarai::script
