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

#ifndef ATHENA_TESTALLOCATOR_H
#define ATHENA_TESTALLOCATOR_H

#include <athena/backend/llvm/BackendAllocator.h>

#include <unordered_map>

class TestAllocator : public athena::backend::llvm::BackendAllocator {
private:
  std::unordered_map<size_t, void*> mAllocationsMap;

public:
  void allocate(const athena::core::inner::Tensor& tensor) override {
    auto mem =
        new unsigned char[tensor.getSize() *
                          athena::core::sizeOfDataType(tensor.getDataType())];
    mAllocationsMap[tensor.getVirtualAddress()] = mem;
  }
  void deallocate(const athena::core::inner::Tensor& tensor) override {
    delete[] static_cast<unsigned char*>(
        mAllocationsMap[tensor.getVirtualAddress()]);
  }
  void* get(const athena::core::inner::Tensor& tensor) override {
    return mAllocationsMap[tensor.getVirtualAddress()];
  }

  void registerDevice(athena::backend::llvm::Device& device) override {}

  void allocate(const athena::core::inner::Tensor& tensor,
                athena::backend::llvm::Device& device) override {
    allocate(tensor);
  }
  void lock(const athena::core::inner::Tensor& tensor,
            athena::backend::llvm::Device& device,
            athena::core::LockType type) override {}

  void lock(const athena::core::inner::Tensor& tensor,
            athena::core::LockType lockType) override {}
  void release(const athena::core::inner::Tensor& tensor) override {}

protected:
  void* getImpl(const athena::core::inner::Tensor& tensor,
                athena::backend::llvm::Device& device) override {
    return get(tensor);
  }
};

#endif // ATHENA_TESTALLOCATOR_H
