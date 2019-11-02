/*
 * Copyright (c) 2019 Athena. All rights reserved.
 * https://getathena.ml
 *
 * Licensed under MIT license.
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an “AS IS” BASIS, WITHOUT
 * WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the
 * License for the specific language governing permissions and limitations under
 * the License.
 */

#include "llvm/ADT/StringRef.h"
#include "llvm/ExecutionEngine/ExecutionEngine.h"
#include "llvm/IRReader/IRReader.h"
#include "llvm/Support/MemoryBuffer.h"
#include "llvm/Support/SourceMgr.h"
#include "llvm/Support/TargetSelect.h"

#include <AthenaJIT.h>
#include <gtest/gtest.h>
#include <memory>

using namespace athena::backend::llvm;

std::string jit_test_add_ir =
    "define i32 @jit_test_add(i32, i32) {\n"
    "  %3 = alloca i32, align 4\n"
    "  %4 = alloca i32, align 4\n"
    "  store i32 %0, i32* %3, align 4\n"
    "  store i32 %1, i32* %4, align 4\n"
    "  %5 = load i32, i32* %3, align 4\n"
    "  %6 = load i32, i32* %4, align 4\n"
    "  %7 = add nsw i32 %5, %6\n"
    "  ret i32 %7\n"
    "}";

class LLVMJITTestType : public ::testing::Test {
    protected:
    std::unique_ptr<AthenaJIT> jitCompiler;

    void SetUp() override {
        ::llvm::InitializeNativeTarget();
        LLVMInitializeNativeAsmPrinter();
        LLVMInitializeNativeAsmParser();
        jitCompiler = AthenaJIT::create();
    }
};

TEST_F(LLVMJITTestType, JITIsAbleToExecuteCode) {
    // Arrange
    std::string irStr = "target datalayout = \"";
    irStr += jitCompiler->getDataLayout().getStringRepresentation();
    irStr += "\"\n";
    irStr += jit_test_add_ir;

    auto irBuffer = llvm::MemoryBuffer::getMemBuffer(irStr);
    llvm::SMDiagnostic err;
    auto ir = llvm::parseIR(*irBuffer, err, jitCompiler->getContext());

    auto jitErr = jitCompiler->addModule(ir);
    if (jitErr) {
        llvm::errs() << jitErr;
        FAIL();
    }

    // Act
    auto sym = jitCompiler->lookup("jit_test_add");
    if (!sym) {
        llvm::consumeError(sym.takeError());
        FAIL();
    }

    auto addFunction = (int (*)(int, int))(intptr_t)sym.get().getAddress();
    auto res = addFunction(5, 4);

    // Assert
    ASSERT_EQ(res, 9);
}
