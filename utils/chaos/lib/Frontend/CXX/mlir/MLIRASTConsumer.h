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

#ifndef ATHENA_MLIRASTCONSUMER_H
#define ATHENA_MLIRASTCONSUMER_H

#include "TypeConverter.h"
#include <clang/AST/ASTConsumer.h>
#include <clang/AST/ASTContext.h>
#include <clang/AST/Mangle.h>
#include <clang/AST/RecursiveASTVisitor.h>
#include <llvm/ADT/ScopedHashTable.h>
#include <llvm/ADT/StringRef.h>
#include <mlir/IR/Builders.h>
#include <mlir/IR/Function.h>
#include <mlir/IR/Module.h>
#include <unordered_map>

namespace chaos {
class MLIRASTConsumer : public clang::ASTConsumer {
protected:
  using VarScope = llvm::ScopedHashTableScope<llvm::StringRef, mlir::Value>;
  using VarTable = llvm::ScopedHashTable<llvm::StringRef, mlir::Value>;
  using TypeScope = llvm::ScopedHashTableScope<llvm::StringRef, mlir::Type>;
  using TypeTable = llvm::ScopedHashTable<llvm::StringRef, mlir::Type>;
  clang::ASTContext& mContext;
  std::unique_ptr<clang::MangleContext> mMangleContext;
  mlir::OwningModuleRef& mMLIRModule;
  mlir::OpBuilder mBuilder;
  TypeConverter mTypeConverter;
  mlir::OpBuilder::InsertPoint mPrevInsertPoint;
  VarTable mSymbolTable;
  TypeTable mTypeTable;
  std::stack<VarScope> mLocalVarScope;
  std::stack<TypeScope> mLocalTypeScope;
  std::unordered_map<std::string, mlir::FuncOp> mFunctionsTable;

  mlir::Value getConstant(clang::APValue* value);

  mlir::Location loc(clang::SourceLocation location);

  std::string mangleName(clang::FunctionDecl* type);

  bool isCanonicalLoop(clang::ForStmt* stmt);

  std::tuple<mlir::Value, llvm::StringRef>
  extractSingleForInit(clang::DeclStmt* stmt);

  mlir::Value extractForUpperBound(clang::BinaryOperator* expr);

  // MARK: Decls

  /// Dispatches decls to appropriate visit calls.
  ///
  /// \param decl is a Clang decl to traverse.
  void visit(clang::Decl* decl);
  void visit(clang::FunctionDecl* functionDecl);
  void visit(clang::TypedefDecl* decl);
  void visit(clang::VarDecl* decl);
  // TODO support for (https://clang.llvm.org/doxygen/classclang_1_1Decl.html):
  //   1. AccessSpecDecl
  //   2. BlockDecl
  //   3. CapturedDecl
  //   4. ClassScopeFunctionSpecializationDecl
  //   5. EmptyDecl
  //   6. ExportDecl
  //   7. ExternCContextDecl
  //   8. FileScopeAsmDecl
  //   9. FriendDecl
  //  10. FriendTemplateDecl
  //  11. ImportDecl - ObjC?
  //  12. LinkageSpecDecl
  //  13. NamedDecl
  //      a. LabelDecl
  //      b. NamespaceAliasDecl
  //      c. NamespaceDecl
  //      d. TemplateDecl
  //      e. TypeDecl
  //      f. UsingDecl
  //      g. UsingDirectiveDecl
  //      h. UsingPackDecl
  //      i. UsingShadowDecl
  //      j. ValueDecl
  //  14. OMPAllocateDecl
  //  15. OMPRequiresDecl
  //  16. OMPThreadPrivateDecl
  //  17. PragmaCommentDecl
  //  18. PragmaDetectMismatchDecl
  //  19. StaticAssertDecl

  // MARK: Statements

  /// Dispatches statements for code generation.
  ///
  /// Expressions are also statements. An expression statement will be
  /// evaluated, the result will be omitted.
  ///
  /// \param stmt is a Clang AST statements to generate code for.
  void visit(clang::Stmt* stmt);
  void visit(clang::ForStmt* loop);
  void visit(clang::DeclStmt* stmt);
  void visit(clang::CompoundAssignOperator* stmt);
  void visit(clang::ReturnStmt* stmt);
  // TODO support (https://clang.llvm.org/doxygen/classclang_1_1Stmt.html):
  //   1. AsmStmt
  //   2. BreakStmt
  //   3. CapturedStmt
  //   4. CompoundStmt
  //   5. ContinueStmt
  //   6. CoreturnStmt
  //   7. CoroutineBodyStmt
  //   8. CXXCatchStmt
  //   9. CXXForRangeStmt
  //  10. CXXTryStmt
  //  11. DoStmt
  //  12. GotoStmt
  //  13. IfStmt
  //  14. IndirectGotoStmt
  //  15. MSDependentExistsStmt
  //  16. NullStmt
  //  17. OMPExecutableDirective and children
  //  18. SEHExceptStmt
  //  19. SEHFinallyStmt
  //  20. SEHLeaveStmt
  //  21. SEHTryStmt
  //  22. SwitchCase
  //  23. SwitchStmt
  //  24. ValueStmt
  //  25. WhileStmt

  // MARK: Expressions

  /// Dispatches expressions for code generation.
  ///
  /// \param expr is a Clang expression to generate code for.
  /// \return is a result value of expression evaluation.
  mlir::Value evaluate(clang::Expr* expr);
  mlir::Value evaluate(clang::BinaryOperator* expr);
  mlir::Value evaluate(clang::UnaryOperator* expr);
  mlir::Value evaluate(clang::FloatingLiteral* expr);
  mlir::Value evaluate(clang::IntegerLiteral* expr);
  mlir::Value evaluate(clang::ImplicitCastExpr* expr);
  mlir::Value evaluate(clang::ArraySubscriptExpr* expr);
  mlir::Value evaluate(clang::DeclRefExpr* expr);
  mlir::Value evaluate(clang::CallExpr* expr);
  mlir::Value evaluate(clang::CXXNewExpr* expr);
  mlir::Value evaluate(clang::CXXDeleteExpr* expr);
  // TODO support for https://clang.llvm.org/doxygen/classclang_1_1Expr.html

public:
  MLIRASTConsumer(clang::ASTContext& ctx, mlir::OwningModuleRef& module)
      : mContext(ctx), mMLIRModule(module), mBuilder(mMLIRModule.get()),
        mTypeConverter(mMLIRModule->getContext()) {
    mMangleContext = std::unique_ptr<clang::MangleContext>(
        clang::ItaniumMangleContext::create(mContext,
                                            mContext.getDiagnostics()));
    mBuilder.setInsertionPointToStart(mMLIRModule->getBody());
  }
  void HandleTranslationUnit(clang::ASTContext& Ctx) override;
};
} // namespace chaos

#endif // ATHENA_MLIRASTCONSUMER_H
