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

// Until we have a proper Graph compiler, this tool can provide a few real
// samples for tests.

#include "mlir/IR/Dialect.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/InitAllDialects.h"
#include "mlir/InitAllPasses.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Support/FileUtilities.h"
#include "llvm/Support/InitLLVM.h"
#include "llvm/Support/SourceMgr.h"

#include "AthenaGraph/AthenaGraphDialect.h"
#include "AthenaGraph/AthenaGraphOps.h"

int main() {
  mlir::registerAllDialects();
  mlir::registerAllPasses();

  mlir::registerDialect<mlir::ath_graph::AthenaGraphDialect>();

  mlir::MLIRContext context;
  mlir::OpBuilder builder(&context);
  mlir::OwningModuleRef module(mlir::ModuleOp::create(builder.getUnknownLoc()));

  builder.setInsertionPointToStart(module->getBody());

  auto athMod =
      builder.create<mlir::ath_graph::ModuleOp>(builder.getUnknownLoc());

  builder.setInsertionPointToStart(&athMod.body().front());

  auto tensorType = mlir::RankedTensorType::get({8}, builder.getF32Type());
  auto inputNodeType = builder.getFunctionType({}, {tensorType});
  auto inpA = builder.create<mlir::ath_graph::NodeOp>(builder.getUnknownLoc(),
                                                      "InputA", inputNodeType);
  {
    mlir::OpBuilder::InsertionGuard guard{builder};
    builder.setInsertionPointToStart(&inpA.getBody().front());

    auto context = inpA.getArgument(inpA.getNumFuncArguments() - 2);
    auto tensor = builder.create<mlir::ath_graph::GetTensor>(
        builder.getUnknownLoc(), context, 1,
        mlir::RankedTensorType::get({8}, builder.getF32Type()));

    builder.create<mlir::ath_graph::AllocOp>(builder.getUnknownLoc(),
                                             tensor.getResult());

    builder.create<mlir::ath_graph::LockOp>(builder.getUnknownLoc(),
                                            builder.getStringAttr("read_write"),
                                            tensor.getResult());
    builder.create<mlir::ath_graph::InvokeLoaderOp>(
        builder.getUnknownLoc(), "MyLoaderLoad", tensor.getResult());

    builder.create<mlir::ath_graph::ReleaseOp>(builder.getUnknownLoc(),
                                               tensor.getResult());
    builder.create<mlir::ath_graph::ReturnOp>(builder.getUnknownLoc(),
                                              tensor.getResult());
  }

  auto inpB = builder.create<mlir::ath_graph::NodeOp>(builder.getUnknownLoc(),
                                                      "InputB", inputNodeType);
  {
    mlir::OpBuilder::InsertionGuard guard{builder};
    builder.setInsertionPointToStart(&inpB.getBody().front());

    auto context = inpB.getArgument(inpB.getNumFuncArguments() - 2);
    auto tensor = builder.create<mlir::ath_graph::GetTensor>(
        builder.getUnknownLoc(), context, 9,
        mlir::RankedTensorType::get({8}, builder.getF32Type()));

    builder.create<mlir::ath_graph::AllocOp>(builder.getUnknownLoc(),
                                             tensor.getResult());

    builder.create<mlir::ath_graph::LockOp>(builder.getUnknownLoc(),
                                            builder.getStringAttr("read_write"),
                                            tensor.getResult());
    builder.create<mlir::ath_graph::InvokeLoaderOp>(
        builder.getUnknownLoc(), "MyLoaderLoad", tensor.getResult());

    builder.create<mlir::ath_graph::ReleaseOp>(builder.getUnknownLoc(),
                                               tensor.getResult());
    builder.create<mlir::ath_graph::ReturnOp>(builder.getUnknownLoc(),
                                              tensor.getResult());
  }

  auto sumNodeType =
      builder.getFunctionType({tensorType, tensorType}, {tensorType});
  auto sumNode = builder.create<mlir::ath_graph::NodeOp>(
      builder.getUnknownLoc(), "SumNode", sumNodeType);

  {
    mlir::OpBuilder::InsertionGuard guard{builder};
    builder.setInsertionPointToStart(&sumNode.getBody().front());

    auto context = sumNode.getArgument(sumNode.getNumFuncArguments() - 2);
    auto tensor = builder.create<mlir::ath_graph::GetTensor>(
        builder.getUnknownLoc(), context, 17, tensorType);

    builder.create<mlir::ath_graph::AllocOp>(builder.getUnknownLoc(),
                                             tensor.getResult());
    builder.create<mlir::ath_graph::LockOp>(builder.getUnknownLoc(),
                                            builder.getStringAttr("read"),
                                            sumNode.getArgument(0));
    builder.create<mlir::ath_graph::LockOp>(builder.getUnknownLoc(),
                                            builder.getStringAttr("read"),
                                            sumNode.getArgument(1));
    builder.create<mlir::ath_graph::LockOp>(builder.getUnknownLoc(),
                                            builder.getStringAttr("read_write"),
                                            tensor.getResult());

    auto unit = builder.create<mlir::ConstantFloatOp>(
        builder.getUnknownLoc(), llvm::APFloat(1.0f), builder.getF32Type());
    builder.create<mlir::ath_graph::AddOp>(
        builder.getUnknownLoc(), tensorType, sumNode.getArgument(0), unit,
        sumNode.getArgument(1), unit, tensor.getResult());

    builder.create<mlir::ath_graph::ReleaseOp>(builder.getUnknownLoc(),
                                               tensor.getResult());
    builder.create<mlir::ath_graph::ReleaseOp>(builder.getUnknownLoc(),
                                               sumNode.getArgument(0));
    builder.create<mlir::ath_graph::ReleaseOp>(builder.getUnknownLoc(),
                                               sumNode.getArgument(1));

    builder.create<mlir::ath_graph::ReturnOp>(builder.getUnknownLoc(),
                                              tensor.getResult());
  }

  auto graphOp = builder.create<mlir::ath_graph::GraphOp>(
      builder.getUnknownLoc(), "SampleGraph");

  {
    mlir::OpBuilder::InsertionGuard guard{builder};
    builder.setInsertionPointToStart(&graphOp.getBody().front());

    auto aRes = builder.create<mlir::ath_graph::EvalOp>(
        builder.getUnknownLoc(), inpA,
        mlir::ValueRange{graphOp.getArgument(0), graphOp.getArgument(1)});

    auto bRes = builder.create<mlir::ath_graph::EvalOp>(
        builder.getUnknownLoc(), inpB,
        mlir::ValueRange{graphOp.getArgument(0), graphOp.getArgument(1)});
    builder.create<mlir::ath_graph::EvalOp>(
        builder.getUnknownLoc(), sumNode,
        mlir::ValueRange{aRes.getResult(0), bRes.getResult(0),
                         graphOp.getArgument(0), graphOp.getArgument(1)});
  }

  module->dump();

  return 0;
}
