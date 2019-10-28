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

#include "common.h"

#include <athena/backend/llvm/LLVMTrivialAllocator.h>
#include <athena/backend/llvm/runtime/builtin.h>
#include <athena/backend/llvm/runtime/structs.h>
#include <athena/core/Context.h>
#include <athena/core/inner/Tensor.h>

#include <cstring>

using namespace athena::backend::llvm;
using namespace athena::backend;
using namespace athena::core::inner;
using namespace athena::core;

// todo move to core
template <typename T>
DataType data_type_cast() {
    new FatalError(-1, "Not implemented");
}

template <>
DataType data_type_cast<float>() {
    return DataType::FLOAT;
}

template <>
DataType data_type_cast<double>() {
    return DataType::DOUBLE;
}

TYPED_TEST(RuntimeTest, GEMMSimple) {
    TypeParam matrix1[] = {1, 1, 2, 2, 3, 3, 4, 4};
    TypeParam matrix2[] = {1, 2, 3, 4, 1, 2, 3, 4};

    Context ctx;

    Tensor a(data_type_cast<TypeParam>(), {2, 4}, ctx);
    Tensor b(data_type_cast<TypeParam>(), {4, 2}, ctx);
    Tensor c(data_type_cast<TypeParam>(), {4, 4}, ctx);

    LLVMTrivialAllocator allocator;
    allocator.allocate(a);
    allocator.allocate(b);
    allocator.allocate(c);

    std::memcpy(reinterpret_cast<void *>(allocator.getRAMPointer(a)), matrix1,
                8 * sizeof(TypeParam));
    std::memcpy(reinterpret_cast<void *>(allocator.getRAMPointer(b)), matrix2,
                8 * sizeof(TypeParam));

    GEMMOptions<TypeParam> opts{false, false, 1, 0};

    gemm<TypeParam>(nullptr, &allocator, &opts, &a, &b, &c);

    auto *cRes = reinterpret_cast<TypeParam *>(allocator.getRAMPointer(c));

    EXPECT_FLOAT_EQ(cRes[0], 2);
    EXPECT_FLOAT_EQ(cRes[1], 4);
    EXPECT_FLOAT_EQ(cRes[2], 6);
    EXPECT_FLOAT_EQ(cRes[3], 8);
    EXPECT_FLOAT_EQ(cRes[4], 4);
    EXPECT_FLOAT_EQ(cRes[5], 8);
    EXPECT_FLOAT_EQ(cRes[6], 12);
    EXPECT_FLOAT_EQ(cRes[7], 16);
    EXPECT_FLOAT_EQ(cRes[8], 6);
    EXPECT_FLOAT_EQ(cRes[9], 12);
    EXPECT_FLOAT_EQ(cRes[10], 18);
    EXPECT_FLOAT_EQ(cRes[11], 24);
    EXPECT_FLOAT_EQ(cRes[12], 8);
    EXPECT_FLOAT_EQ(cRes[13], 16);
    EXPECT_FLOAT_EQ(cRes[14], 24);
    EXPECT_FLOAT_EQ(cRes[15], 32);
}

TYPED_TEST(RuntimeTest, GEMMTransposeA) {
    TypeParam matrix[] = {1, 2, 3, 4, 1, 2, 3, 4};

    Context ctx;

    Tensor a(data_type_cast<TypeParam>(), {4, 2}, ctx);
    Tensor b(data_type_cast<TypeParam>(), {4, 2}, ctx);
    Tensor c(data_type_cast<TypeParam>(), {4, 4}, ctx);

    LLVMTrivialAllocator allocator;
    allocator.allocate(a);
    allocator.allocate(b);
    allocator.allocate(c);

    std::memcpy(reinterpret_cast<void *>(allocator.getRAMPointer(a)), matrix,
                8 * sizeof(TypeParam));
    std::memcpy(reinterpret_cast<void *>(allocator.getRAMPointer(b)), matrix,
                8 * sizeof(TypeParam));

    GEMMOptions<TypeParam> opts{true, false, 1, 0};

    gemm<TypeParam>(nullptr, &allocator, &opts, &a, &b, &c);

    auto *cRes = reinterpret_cast<TypeParam *>(allocator.getRAMPointer(c));

    EXPECT_FLOAT_EQ(cRes[0], 2);
    EXPECT_FLOAT_EQ(cRes[1], 4);
    EXPECT_FLOAT_EQ(cRes[2], 6);
    EXPECT_FLOAT_EQ(cRes[3], 8);
    EXPECT_FLOAT_EQ(cRes[4], 4);
    EXPECT_FLOAT_EQ(cRes[5], 8);
    EXPECT_FLOAT_EQ(cRes[6], 12);
    EXPECT_FLOAT_EQ(cRes[7], 16);
    EXPECT_FLOAT_EQ(cRes[8], 6);
    EXPECT_FLOAT_EQ(cRes[9], 12);
    EXPECT_FLOAT_EQ(cRes[10], 18);
    EXPECT_FLOAT_EQ(cRes[11], 24);
    EXPECT_FLOAT_EQ(cRes[12], 8);
    EXPECT_FLOAT_EQ(cRes[13], 16);
    EXPECT_FLOAT_EQ(cRes[14], 24);
    EXPECT_FLOAT_EQ(cRes[15], 32);
}

TYPED_TEST(RuntimeTest, GEMMTransposeB) {
    TypeParam matrix[] = {1, 1, 2, 2, 3, 3, 4, 4};

    Context ctx;

    Tensor a(data_type_cast<TypeParam>(), {2, 4}, ctx);
    Tensor b(data_type_cast<TypeParam>(), {2, 4}, ctx);
    Tensor c(data_type_cast<TypeParam>(), {4, 4}, ctx);

    LLVMTrivialAllocator allocator;
    allocator.allocate(a);
    allocator.allocate(b);
    allocator.allocate(c);

    std::memcpy(reinterpret_cast<void *>(allocator.getRAMPointer(a)), matrix,
                8 * sizeof(TypeParam));
    std::memcpy(reinterpret_cast<void *>(allocator.getRAMPointer(b)), matrix,
                8 * sizeof(TypeParam));

    GEMMOptions<TypeParam> opts{false, true, 1, 0};

    gemm<TypeParam>(nullptr, &allocator, &opts, &a, &b, &c);

    auto *cRes = reinterpret_cast<TypeParam *>(allocator.getRAMPointer(c));

    EXPECT_FLOAT_EQ(cRes[0], 2);
    EXPECT_FLOAT_EQ(cRes[1], 4);
    EXPECT_FLOAT_EQ(cRes[2], 6);
    EXPECT_FLOAT_EQ(cRes[3], 8);
    EXPECT_FLOAT_EQ(cRes[4], 4);
    EXPECT_FLOAT_EQ(cRes[5], 8);
    EXPECT_FLOAT_EQ(cRes[6], 12);
    EXPECT_FLOAT_EQ(cRes[7], 16);
    EXPECT_FLOAT_EQ(cRes[8], 6);
    EXPECT_FLOAT_EQ(cRes[9], 12);
    EXPECT_FLOAT_EQ(cRes[10], 18);
    EXPECT_FLOAT_EQ(cRes[11], 24);
    EXPECT_FLOAT_EQ(cRes[12], 8);
    EXPECT_FLOAT_EQ(cRes[13], 16);
    EXPECT_FLOAT_EQ(cRes[14], 24);
    EXPECT_FLOAT_EQ(cRes[15], 32);
}

TYPED_TEST(RuntimeTest, GEMMTransposeBoth) {
    TypeParam matrix1[] = {1, 1, 2, 2, 3, 3, 4, 4};
    TypeParam matrix2[] = {1, 2, 3, 4, 1, 2, 3, 4};

    Context ctx;

    Tensor a(data_type_cast<TypeParam>(), {4, 2}, ctx);
    Tensor b(data_type_cast<TypeParam>(), {2, 4}, ctx);
    Tensor c(data_type_cast<TypeParam>(), {4, 4}, ctx);

    LLVMTrivialAllocator allocator;
    allocator.allocate(a);
    allocator.allocate(b);
    allocator.allocate(c);

    std::memcpy(reinterpret_cast<void *>(allocator.getRAMPointer(a)), matrix2,
                8 * sizeof(TypeParam));
    std::memcpy(reinterpret_cast<void *>(allocator.getRAMPointer(b)), matrix1,
                8 * sizeof(TypeParam));

    GEMMOptions<TypeParam> opts{true, true, 1, 0};

    gemm<TypeParam>(nullptr, &allocator, &opts, &a, &b, &c);

    auto *cRes = reinterpret_cast<TypeParam *>(allocator.getRAMPointer(c));

    EXPECT_FLOAT_EQ(cRes[0], 2);
    EXPECT_FLOAT_EQ(cRes[1], 4);
    EXPECT_FLOAT_EQ(cRes[2], 6);
    EXPECT_FLOAT_EQ(cRes[3], 8);
    EXPECT_FLOAT_EQ(cRes[4], 4);
    EXPECT_FLOAT_EQ(cRes[5], 8);
    EXPECT_FLOAT_EQ(cRes[6], 12);
    EXPECT_FLOAT_EQ(cRes[7], 16);
    EXPECT_FLOAT_EQ(cRes[8], 6);
    EXPECT_FLOAT_EQ(cRes[9], 12);
    EXPECT_FLOAT_EQ(cRes[10], 18);
    EXPECT_FLOAT_EQ(cRes[11], 24);
    EXPECT_FLOAT_EQ(cRes[12], 8);
    EXPECT_FLOAT_EQ(cRes[13], 16);
    EXPECT_FLOAT_EQ(cRes[14], 24);
    EXPECT_FLOAT_EQ(cRes[15], 32);
}