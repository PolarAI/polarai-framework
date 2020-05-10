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

#include <effcee/effcee.h>
#include <fstream>
#include <gtest/gtest.h>

#include <athena/backend/llvm/mlir/LLVMTranslator.h>
#include <llvm/IR/LLVMContext.h>
#include <llvm/IR/Module.h>
#include <llvm/Support/raw_os_ostream.h>
#include <mlir/IR/Builders.h>
#include <mlir/IR/Dialect.h>
#include <mlir/IR/Module.h>
#include <mlir/Parser.h>

const std::string sampleIr =
    "module {\n"
    "  func @MainGraph() {\n"
    "    %0 = \"graph.alloca\"() {cluster_id = 0 : i64, node_id = 0 : i64, "
    "node_name = \"inpA\", tensor_addr = 4 : i64} : () -> tensor<3xf32>\n"
    "    %1 = \"graph.alloca\"() {cluster_id = 0 : i64, node_id = 1 : i64, "
    "node_name = \"inpB\", tensor_addr = 1 : i64} : () -> tensor<3xf32>\n"
    "    %2 = \"graph.alloca\"() {cluster_id = 0 : i64, node_id = 2 : i64, "
    "node_name = \"add\", tensor_addr = 7 : i64} : () -> tensor<3xf32>\n"
    "    %3 = \"graph.add\"(%1, %0) {cluster_id = 0 : i64, node_id = 2 : i64, "
    "node_name = \"add\", tensor_addr = 7 : i64} : (tensor<3xf32>, "
    "tensor<3xf32>) -> tensor<3xf32>\n"
    "    \"graph.return\"() : () -> ()\n"
    "  }\n"
    "}";

const std::string matches =
    "CHECK: define void @evaluateMainGraph()\n"
    "CHECK: call void @inpA()\n"
    "CHECK: call void @inpB()\n"
    "CHECK: call void @add()\n"
    "CHECK: define void @inpA() !node_id !0 !node_name !1 !cluster_id !0\n"
    "CHECK: %0 = call i64* @getDeviceForNode(i64 0)\n"
    "CHECK: %1 = call i64* @getAllocator()\n"
    "CHECK: %2 = call i64* @getTensorPtr(i64 4)\n"
    "CHECK: define void @add() !node_id !4 !node_name !5 !cluster_id !4\n"
    "CHECK: %0 = call i64* @getDeviceForNode(i64 2)\n"
    "CHECK: %1 = call i64* @getAllocator()\n"
    "CHECK: %2 = call i64* @getTensorPtr(i64 7)\n"
    "CHECK: call void @allocate(i64* %0, i64* %1, i64* %2)\n"
    "CHECK: %3 = call i64* @getDeviceForNode(i64 2)\n"
    "CHECK: %4 = call i64* @getAllocator()\n"
    "CHECK: %5 = call i64* @getTensorPtr(i64 1)\n"
    "CHECK: %6 = call i64* @getTensorPtr(i64 4)\n"
    "CHECK: call void @athn_add_f(i64* %3, i64* %4, i64* %5, i64* %6)\n"
    "CHECK: ret void";

TEST(MLIRRegression, DISABLED_TranslatesCorrectlyToLLVM) {
  llvm::LLVMContext llvmContext;
  llvm::Module llvmModule("test", llvmContext);

  mlir::MLIRContext mlirContext;
  mlir::OwningModuleRef mlirModule =
      mlir::parseSourceString(sampleIr, &mlirContext);
  athena::backend::llvm::LLVMTranslator translator(mlirModule, llvmModule);
  translator.translate();
  std::string str;
  llvm::raw_string_ostream stringOstream(str);
  llvmModule.print(stringOstream, nullptr);
  stringOstream.str();

  auto result =
      effcee::Match(str, matches, effcee::Options().SetChecksName("checks"));

  if (result) {
    SUCCEED();
  } else {
    // Otherwise, you can get a status code and a detailed message.
    switch (result.status()) {
    case effcee::Result::Status::NoRules:
      std::cout << "error: Expected check rules\n";
      break;
    case effcee::Result::Status::Fail:
      std::cout << "The input failed to match check rules:\n";
      break;
    default:
      break;
    }
    std::cout << result.message() << std::endl;
    FAIL();
  }
}