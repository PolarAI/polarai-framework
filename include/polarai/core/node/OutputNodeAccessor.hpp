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

#include <polarai/core/loader/AbstractLoader.hpp>
#include <polarai/core/tensor/Accessor.hpp>

namespace polarai::core {
template <typename T> class OutputNodeAccessor : public Accessor<T> {
protected:
  using Accessor<T>::linearIndex;
public:
  OutputNodeAccessor(internal::TensorAllocator& allocator,
                     internal::TensorInternal& tensor)
      : mAllocator(&allocator), mTensor(&tensor) {
    mAllocator->lock(*mTensor, internal::LockType::READ_WRITE);
    mData = static_cast<T*>(mAllocator->get(*mTensor));
  }

  OutputNodeAccessor(const OutputNodeAccessor<T>&) = delete;
  auto operator=(const OutputNodeAccessor<T>) = delete;

  ~OutputNodeAccessor() { mAllocator->release(*mTensor); }

  auto operator()(std::initializer_list<size_t> idx) -> T& override {
    return mData[linearIndex(idx, mTensor->getShape().getShape())];
  }

  auto operator()(size_t idx) -> T& override { return mData[idx]; }

  auto getShape() -> const std::vector<size_t>& override {
    return mTensor->getShape().getShape();
  }

  auto getRawPtr() -> T* override { return mData; }

  void release() { mAllocator->release(*mTensor); }

private:
  internal::TensorAllocator* mAllocator;
  internal::TensorInternal* mTensor;
  T* mData;
};
} // namespace polarai::core
