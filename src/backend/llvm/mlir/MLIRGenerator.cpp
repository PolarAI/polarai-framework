/*
 * Copyright (c) 2019 Athena. All rights reserved.
 * https://getathena.ml
 *
 * Licensed under MIT license.
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS, WITHOUT
 * WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the
 * License for the specific language governing permissions and limitations under
 * the License.
 */

#include "GraphDialect.h"

#include <athena/backend/llvm/mlir/MLIRGenerator.h>

using namespace athena::backend::llvm;
using namespace athena::core;
using namespace athena::core::inner;

void MLIRGenerator::openNode(std::string_view name) {}
void MLIRGenerator::closeNode() {}
void MLIRGenerator::generateFunctionHeader(const std::string& name) {}
void MLIRGenerator::generateFunctionFooter() {
    mBuilder.create<ReturnOp>(mBuilder.getUnknownLoc());
}
void MLIRGenerator::generateLoad(const AbstractLoader& loader, Tensor& tensor) {
}
void MLIRGenerator::generateImpl(std::string& string, Tensor& a) {
    mlir::Value val = mBuilder.create<AllocaOp>(mBuilder.getUnknownLoc(), a, 1,
                                                "temp_node_name", 0);
    mTensorValueMap[a.getVirtualAddress()] = val;
}
void MLIRGenerator::generateImpl(std::string& string, Tensor& a, void*& b) {}
void MLIRGenerator::generateImpl(std::string& string, Tensor& a, Tensor& b) {}
void MLIRGenerator::generateImpl(std::string& name,
                                 Tensor& a,
                                 Tensor& b,
                                 Tensor& c) {
    if (name == "add") {
        assert(mTensorValueMap.count(a.getVirtualAddress()));
        assert(mTensorValueMap.count(b.getVirtualAddress()));
        auto aVal = mTensorValueMap[a.getVirtualAddress()];
        auto bVal = mTensorValueMap[b.getVirtualAddress()];

        mlir::Value cVal = mBuilder.create<AddOp>(
            mBuilder.getUnknownLoc(), aVal, bVal, c, 1, "temp_node_name", 0);
        mTensorValueMap[c.getVirtualAddress()] = cVal;
    }
}
void MLIRGenerator::generateImpl(std::string& string,
                                 Tensor& a,
                                 uint64_t scaleA,
                                 Tensor& b,
                                 uint64_t scaleB,
                                 Tensor& c) {}
void MLIRGenerator::generateImpl(
    std::string& string, void* pVoid, Tensor& a, Tensor& b, Tensor& c) {}

MLIRGenerator::MLIRGenerator() : mBuilder(&mContext) {
    mModule = mlir::ModuleOp::create(mBuilder.getUnknownLoc());
    mlir::FuncOp function = mlir::FuncOp::create(
        mBuilder.getUnknownLoc(), "evaluate",
        mlir::FunctionType::get({}, llvm::None, &mContext));
    auto& entryBlock = *function.addEntryBlock();
    mBuilder.setInsertionPointToStart(&entryBlock);
    mModule.push_back(function);
}
