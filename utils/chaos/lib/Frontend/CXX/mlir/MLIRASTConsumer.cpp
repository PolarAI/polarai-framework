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

#include "MLIRASTConsumer.h"
#include <Dialects/ClangDialect.h>
#include <clang/AST/AST.h>
#include <clang/AST/Decl.h>
#include <iostream>
#include <mlir/Dialect/AffineOps/AffineOps.h>
#include <mlir/Dialect/StandardOps/Ops.h>
#include <mlir/IR/Function.h>

namespace chaos {
void MLIRASTConsumer::HandleTranslationUnit(clang::ASTContext& Ctx) {
  auto traversalScope = mContext.getTraversalScope();

  for (auto* decl : traversalScope) {
    visit(decl);
  }
}

void MLIRASTConsumer::visit(clang::Decl* decl) {
  if (!decl) {
    llvm_unreachable("Broken decl");
  }

  if (auto* trUnitDecl =
          llvm::dyn_cast_or_null<clang::TranslationUnitDecl>(decl)) {
    for (auto it = trUnitDecl->decls_begin(); it != trUnitDecl->decls_end();
         ++it) {
      visit(*it);
    }
    return;
  } else if (decl->isFunctionOrFunctionTemplate()) {
    visit(decl->getAsFunction());
    return;
  }

#define dispatch(type, d)                                                      \
  if (auto c = llvm::dyn_cast_or_null<type>(d)) {                              \
    visit(c);                                                                  \
    return;                                                                    \
  }

  dispatch(clang::TypedefDecl, decl);
  dispatch(clang::VarDecl, decl);

#undef dispatch

  llvm::errs() << "Unknown statement: " << decl->getDeclKindName() << "\n";
  decl->dump(llvm::errs());
  llvm_unreachable(".");
}
void MLIRASTConsumer::visit(clang::FunctionDecl* functionDecl) {
  std::string mangledName = mangleName(functionDecl);

  if (functionDecl->isTemplated()) {
    // todo instantiate templates instead
    std::cout << "skip templates\n";
    return;
  }

  auto funcType = functionDecl->getFunctionType();

  auto mlirType = mTypeConverter.convert(*funcType);

  llvm::SmallVector<mlir::NamedAttribute, 3> attrs;
  auto funcOp = mBuilder.create<mlir::FuncOp>(mBuilder.getUnknownLoc(),
                                              mangledName, mlirType, attrs);

  // todo check for duplicates
  mFunctionsTable.insert({mangledName, funcOp});

  mPrevInsertPoint = mBuilder.saveInsertionPoint();

  if (functionDecl->hasBody()) {
    mBuilder.setInsertionPointToStart(funcOp.addEntryBlock());
    mLocalVarScope.emplace(mSymbolTable);
    for (size_t idx = 0; idx < funcOp.getNumArguments(); idx++) {
      mSymbolTable.insert(functionDecl->getParamDecl(idx)->getName(),
                          funcOp.getArgument(idx));
    }
    visit(functionDecl->getBody());
    mLocalVarScope.pop();
  }

  // fixme always add terminator in the end

  mBuilder.restoreInsertionPoint(mPrevInsertPoint);
}
void MLIRASTConsumer::visit(clang::Stmt* stmt) {
  if (!stmt) {
    llvm_unreachable("Invalid statement");
  }
  if (auto c = llvm::dyn_cast_or_null<clang::CompoundStmt>(stmt)) {
    for (auto it = c->body_begin(); it != c->body_end(); ++it) {
      visit(*it);
    }
    return;
  }

#define dispatch(type, d)                                                      \
  if (auto c = llvm::dyn_cast_or_null<type>(d)) {                              \
    visit(c);                                                                  \
    return;                                                                    \
  }

  dispatch(clang::ForStmt, stmt);
  dispatch(clang::DeclStmt, stmt);
  dispatch(clang::CompoundAssignOperator, stmt);
  dispatch(clang::ReturnStmt, stmt);

#undef dispatch

  // Expressions are also statements.
  if (auto c = llvm::dyn_cast_or_null<clang::Expr>(stmt)) {
    evaluate(c);
    return;
  }

  llvm::errs() << "Unknown statement: " << stmt->getStmtClassName() << "\n";
  stmt->dump(llvm::errs());
  stmt->dumpPretty(mContext);
  llvm_unreachable(".");
}

void MLIRASTConsumer::visit(clang::ForStmt* loop) {
  auto clangLoop = mBuilder.create<clang::ForOp>(loc(loop->getForLoc()));

  auto indVars =
      buildForInit(loop->getInit(), clangLoop.getInit(), clangLoop.getCond());
  buildForCond(loop->getCond(), clangLoop.getCond(), clangLoop.getBody(),
               indVars);
  auto checkPoint = mBuilder.saveInsertionPoint();
  mLocalVarScope.emplace(mSymbolTable);
  for (size_t idx = 0; idx < indVars.size(); idx++) {
    mSymbolTable.insert(indVars[idx], clangLoop.getBody()->getArgument(idx));
  }
  mBuilder.setInsertionPointToStart(clangLoop.getBody());
  visit(loop->getBody());

  llvm::SmallVector<mlir::Value, 1> indVarValues;
  for (auto& indVar : indVars) {
    auto var = mSymbolTable.lookup(indVar);
    clangLoop.getInc()->addArgument(var.getType());
    indVarValues.push_back(var);
  }

  mBuilder.create<mlir::BranchOp>(mBuilder.getUnknownLoc(), clangLoop.getInc());

  mLocalVarScope.pop();

  buildForInc(loop->getInc(), clangLoop.getInc(), clangLoop.getCond(), indVars);

  mBuilder.restoreInsertionPoint(checkPoint);
}
void MLIRASTConsumer::visit(clang::DeclStmt* stmt) {
  for (auto* decl : stmt->getDeclGroup()) {
    visit(decl);
  }
}
void MLIRASTConsumer::visit(clang::TypedefDecl* decl) {}

mlir::Value MLIRASTConsumer::getConstant(clang::APValue* value) {
  if (value->isInt()) {
    return mBuilder.create<mlir::ConstantIntOp>(mBuilder.getUnknownLoc(),
                                                value->getInt().getZExtValue(),
                                                value->getInt().getBitWidth());
  }
  llvm_unreachable("err");
}

void MLIRASTConsumer::visit(clang::VarDecl* decl) {
  if (auto val = decl->evaluateValue()) {
    auto mlirVal = getConstant(val);
    mSymbolTable.insert(decl->getName(), mlirVal);
  } else {
    auto init = evaluate(decl->getInit());
    mSymbolTable.insert(decl->getName(), init);
  }
}
void MLIRASTConsumer::visit(clang::CompoundAssignOperator* stmt) {
  auto lhs = evaluate(stmt->getLHS());
  auto rhs = evaluate(stmt->getRHS());

  mlir::Value opRes;
  switch (stmt->getOpcode()) {
  case clang::BinaryOperatorKind::BO_AddAssign:
    if (lhs->getType().isa<mlir::IntegerType>()) {
      opRes = mBuilder.create<mlir::AddIOp>(loc(stmt->getBeginLoc()), lhs, rhs);
    } else if (lhs->getType().isa<mlir::FloatType>()) {
      opRes = mBuilder.create<mlir::AddFOp>(loc(stmt->getBeginLoc()), lhs, rhs);
    }
    break;
  default:
    stmt->dump();
    llvm_unreachable("Unsupported expr");
  }

  if (auto subscript =
          llvm::dyn_cast_or_null<clang::ArraySubscriptExpr>(stmt->getLHS())) {
    auto var = evaluate(subscript->getLHS());
    auto sub = evaluate(subscript->getRHS());

    auto idx = mBuilder.create<mlir::IndexCastOp>(loc(stmt->getBeginLoc()), sub,
                                                  mBuilder.getIndexType());

    mlir::ValueRange range{idx};

    mBuilder.create<mlir::StoreOp>(loc(stmt->getBeginLoc()), opRes, var, range);
  } else {
    // fixme check if this is true
    mBuilder.create<mlir::StoreOp>(loc(stmt->getBeginLoc()), opRes, lhs);
  }
}
mlir::Value MLIRASTConsumer::evaluate(clang::Expr* expr) {
#define dispatch(type, d)                                                      \
  if (auto c = llvm::dyn_cast_or_null<type>(d)) {                              \
    return evaluate(c);                                                        \
  }

  dispatch(clang::FloatingLiteral, expr);
  dispatch(clang::IntegerLiteral, expr);
  dispatch(clang::ArraySubscriptExpr, expr);
  dispatch(clang::ImplicitCastExpr, expr);
  dispatch(clang::CallExpr, expr);
  dispatch(clang::UnaryOperator, expr);
  dispatch(clang::BinaryOperator, expr);
  dispatch(clang::DeclRefExpr, expr);
  dispatch(clang::CXXNewExpr, expr);
  dispatch(clang::CXXDeleteExpr, expr);

#undef dispatch

  llvm::errs() << "Unknown expression: " << expr->getStmtClassName() << "\n";
  expr->dump(llvm::errs());
  expr->dumpPretty(mContext);
  llvm_unreachable(".");
}
mlir::Value MLIRASTConsumer::evaluate(clang::BinaryOperator* expr) {
  auto lhs = evaluate(expr->getLHS());
  auto rhs = evaluate(expr->getRHS());

  if (lhs->getType() != rhs.getType()) {
    if (lhs->getType().isa<mlir::IndexType>()) {
      lhs = mBuilder.create<mlir::IndexCastOp>(loc(expr->getExprLoc()), lhs,
                                               rhs.getType());
    } else if (rhs->getType().isa<mlir::IndexType>()) {
      rhs = mBuilder.create<mlir::IndexCastOp>(loc(expr->getExprLoc()), rhs,
                                               lhs.getType());
    } else {
      expr->dump();
      lhs->getType().dump();
      rhs->getType().dump();
      llvm_unreachable("Types are expected to match");
    }
  }

  mlir::Value res;

  switch (expr->getOpcode()) {
  case clang::BinaryOperatorKind::BO_Add:
    if (lhs->getType().isa<mlir::IntegerType>()) {
      res = mBuilder.create<mlir::AddIOp>(loc(expr->getBeginLoc()), lhs, rhs);
    } else if (lhs->getType().isa<mlir::FloatType>()) {
      res = mBuilder.create<mlir::AddFOp>(loc(expr->getBeginLoc()), lhs, rhs);
    } else {
      llvm_unreachable("Unsupported types");
    }
    break;
  case clang::BinaryOperatorKind::BO_Mul:
    if (lhs->getType().isa<mlir::IntegerType>()) {
      res = mBuilder.create<mlir::MulIOp>(loc(expr->getBeginLoc()), lhs, rhs);
    } else if (lhs->getType().isa<mlir::FloatType>()) {
      res = mBuilder.create<mlir::MulFOp>(loc(expr->getBeginLoc()), lhs, rhs);
    } else {
      llvm_unreachable("Unsupported types");
    }
    break;
  case clang::BinaryOperatorKind::BO_LT:
    // fixme correct opcode
    if (lhs->getType().isa<mlir::IntegerType>()) {
      res = mBuilder.create<mlir::CmpIOp>(loc(expr->getExprLoc()),
                                          mlir::CmpIPredicate::slt, lhs, rhs);
    } else {
      llvm_unreachable("Unsupported types");
    }
    break;
  default:
    expr->dump();
    llvm_unreachable("Err");
  }

  return res;
}
mlir::Location MLIRASTConsumer::loc(clang::SourceLocation location) {
  return mBuilder.getUnknownLoc();
}
mlir::Value MLIRASTConsumer::evaluate(clang::FloatingLiteral* expr) {
  auto type = mTypeConverter.convert(expr->getType());
  return mBuilder.create<mlir::ConstantFloatOp>(
      loc(expr->getBeginLoc()), expr->getValue(), type.cast<mlir::FloatType>());
}
mlir::Value MLIRASTConsumer::evaluate(clang::ImplicitCastExpr* expr) {
  if (expr->getCastKind() == clang::CastKind::CK_LValueToRValue ||
      expr->getCastKind() == clang::CastKind::CK_NoOp) {
    // fixme is special treatment needed?
    return evaluate(expr->getSubExpr());
  } else if (expr->getCastKind() == clang::CastKind::CK_IntegralCast) {
    // fixme introduce missing cast op
    return evaluate(expr->getSubExpr());
  }
  expr->dump();
  llvm_unreachable("Unsupported cast");
}
mlir::Value MLIRASTConsumer::evaluate(clang::ArraySubscriptExpr* expr) {
  auto lhs = evaluate(expr->getLHS());
  auto rhs = evaluate(expr->getRHS());
  auto idx = mBuilder.create<mlir::IndexCastOp>(rhs.getLoc(), rhs,
                                                mBuilder.getIndexType());

  mlir::ValueRange range{idx};

  return mBuilder.create<mlir::LoadOp>(loc(expr->getBeginLoc()), lhs, range);
}
mlir::Value MLIRASTConsumer::evaluate(clang::DeclRefExpr* expr) {
  if (mSymbolTable.count(expr->getDecl()->getName()))
    return mSymbolTable.lookup(expr->getDecl()->getName());

  llvm::errs() << "Var not found.\n";
  llvm::errs() << "Requested " << expr->getDecl()->getName() << "\n";
  llvm::errs() << "Available: ";
  llvm_unreachable("Var not found");
}
mlir::Value MLIRASTConsumer::evaluate(clang::CallExpr* expr) {
  if (auto func = expr->getDirectCallee()) {
    auto mangledName = mangleName(func);
    llvm::SmallVector<mlir::Value, 5> args;
    for (size_t idx = 0; idx < expr->getNumArgs(); idx++) {
      args.push_back(evaluate(expr->getArg(idx)));
    }

    auto call = mBuilder.create<mlir::CallOp>(
        loc(expr->getExprLoc()), mFunctionsTable[mangledName], args);
    // C++ functions return only one value.
    return call.getResult(0);
  }

  expr->dump();
  llvm_unreachable("Unsupported call expr");
}
std::string MLIRASTConsumer::mangleName(clang::FunctionDecl* type) {
  std::string mangledName;
  if (mMangleContext->shouldMangleDeclName(type)) {
    llvm::raw_string_ostream mangledStream(mangledName);
    mMangleContext->mangleName(type, mangledStream);
    mangledStream.flush();
  } else {
    mangledName = type->getName();
  }
  return mangledName;
}
mlir::Value MLIRASTConsumer::evaluate(clang::CXXNewExpr* expr) {
  auto resType = expr->getAllocatedType();
  if (resType->isBuiltinType()) {
    mlir::Value size;
    if (expr->isArray()) {
      size = evaluate(expr->getArraySize().getValue());
    } else {
      size =
          mBuilder.create<mlir::ConstantIntOp>(mBuilder.getUnknownLoc(), 1, 64);
    }

    auto memrefType = mTypeConverter.convert(expr->getAllocatedType());
    // fixme get proper alignment
    auto alignment = mBuilder.getI64IntegerAttr(4);
    return mBuilder.create<mlir::AllocOp>(loc(expr->getExprLoc()), memrefType,
                                          size, alignment);
  } else {
    llvm_unreachable("Unsupported new");
  }
}
mlir::Value MLIRASTConsumer::evaluate(clang::IntegerLiteral* expr) {
  auto type = mTypeConverter.convert(expr->getType());
  return mBuilder.create<mlir::ConstantIntOp>(loc(expr->getBeginLoc()),
                                              *expr->getValue().getRawData(),
                                              type.cast<mlir::IntegerType>());
}
mlir::Value MLIRASTConsumer::evaluate(clang::CXXDeleteExpr* expr) {
  auto resType = expr->getDestroyedType();
  if (resType->isBuiltinType()) {
    auto memrefVal = evaluate(expr->getArgument());
    mBuilder.create<mlir::DeallocOp>(loc(expr->getExprLoc()), memrefVal);
    return mlir::Value();
  } else {
    llvm_unreachable("Unsupported delete");
  }
}
void MLIRASTConsumer::visit(clang::ReturnStmt* stmt) {
  if (auto retExpr = stmt->getRetValue()) {
    auto retValue = evaluate(retExpr);
    mBuilder.create<mlir::ReturnOp>(loc(stmt->getReturnLoc()), retValue);
  } else {
    mBuilder.create<mlir::ReturnOp>(loc(stmt->getReturnLoc()));
  }
}
llvm::SmallVector<llvm::StringRef, 1>
MLIRASTConsumer::buildForInit(clang::Stmt* initStmt, mlir::Block* initBlock,
                              mlir::Block* condBlock) {
  llvm::SmallVector<llvm::StringRef, 1> varNames;
  llvm::SmallVector<mlir::Value, 1> initValues;

  auto checkPoint = mBuilder.saveInsertionPoint();
  mBuilder.setInsertionPointToStart(initBlock);

  if (auto declStmt = llvm::dyn_cast_or_null<clang::DeclStmt>(initStmt)) {
    if (auto varDecl =
            llvm::dyn_cast_or_null<clang::VarDecl>(declStmt->getSingleDecl())) {
      initValues.push_back(evaluate(varDecl->getInit()));
      varNames.push_back(varDecl->getName());
    } else {
      initStmt->dump();
      llvm_unreachable("Unsupported For init statement");
    }
  } else {
    initStmt->dump();
    llvm_unreachable("Unsupported For init statement");
  }

  for (auto& arg : initValues) {
    condBlock->addArgument(arg.getType());
  }

  mBuilder.create<mlir::BranchOp>(loc(initStmt->getEndLoc()), condBlock,
                                  initValues);

  mBuilder.restoreInsertionPoint(checkPoint);

  return varNames;
}
void MLIRASTConsumer::buildForCond(clang::Expr* condExpr,
                                   mlir::Block* condBlock,
                                   mlir::Block* bodyBlock,
                                   llvm::ArrayRef<llvm::StringRef> indVars) {
  mLocalVarScope.emplace(mSymbolTable);
  for (size_t idx = 0; idx < indVars.size(); idx++) {
    mSymbolTable.insert(indVars[idx], condBlock->getArgument(idx));
  }

  auto checkPoint = mBuilder.saveInsertionPoint();
  mBuilder.setInsertionPointToStart(condBlock);

  auto condExprVal = evaluate(condExpr);

  llvm::SmallVector<mlir::Value, 1> vars;

  for (auto indVar : indVars) {
    auto var = mSymbolTable.lookup(indVar);
    bodyBlock->addArgument(var.getType());
    vars.push_back(var);
  }

  // todo special for loop terminator
  mBuilder.create<mlir::BranchOp>(loc(condExpr->getExprLoc()), bodyBlock, vars);

  mBuilder.restoreInsertionPoint(checkPoint);
  mLocalVarScope.pop();
}
void MLIRASTConsumer::buildForInc(clang::Expr* incExpr, mlir::Block* incBlock,
                                  mlir::Block* condBlock,
                                  llvm::ArrayRef<llvm::StringRef> indVars) {
  auto checkPoint = mBuilder.saveInsertionPoint();
  mLocalVarScope.emplace(mSymbolTable);

  for (size_t idx = 0; idx < indVars.size(); idx++) {
    mSymbolTable.insert(indVars[idx], incBlock->getArgument(idx));
  }

  mBuilder.setInsertionPointToStart(incBlock);
  visit(incExpr);

  llvm::SmallVector<mlir::Value, 1> args;
  for (auto& indVar : indVars) {
    args.push_back(mSymbolTable.lookup(indVar));
  }

  mBuilder.create<mlir::BranchOp>(mBuilder.getUnknownLoc(), condBlock, args);

  mLocalVarScope.pop();
  mBuilder.restoreInsertionPoint(checkPoint);
}
mlir::Value MLIRASTConsumer::evaluate(clang::UnaryOperator* expr) {
  mlir::Value res;
  switch (expr->getOpcode()) {
  case clang::UnaryOperatorKind::UO_PostInc: {
    auto oldVal = evaluate(expr->getSubExpr());
    mlir::Value unit;
    if (oldVal.getType().isa<mlir::IntegerType>()) {
      auto type = oldVal.getType().cast<mlir::IntegerType>();
      unit = mBuilder.create<mlir::ConstantIntOp>(mBuilder.getUnknownLoc(), 1,
                                                  type.getWidth());
    } else if (oldVal.getType().isa<mlir::IndexType>()) {
      unit =
          mBuilder.create<mlir::ConstantIndexOp>(mBuilder.getUnknownLoc(), 1);
    } else {
      llvm_unreachable("Wrong type");
    }
    res = mBuilder.create<mlir::AddIOp>(loc(expr->getExprLoc()), oldVal, unit);

    if (auto declRef =
            llvm::dyn_cast_or_null<clang::DeclRefExpr>(expr->getSubExpr())) {
      auto v = mSymbolTable.begin(declRef->getNameInfo().getAsString());
      *v = res;
    } else {
      llvm_unreachable("unexpected");
    }
  } break;
  default:
    expr->dump();
    llvm_unreachable("Unsupported unary expression");
  }
  return res;
}
} // namespace chaos