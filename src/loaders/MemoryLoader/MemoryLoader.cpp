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

#include <athena/loaders/MemoryLoader/MemoryLoader.h>

#include <cassert>
#include <cstring>

namespace athena::loaders {

MemoryLoader::MemoryLoader(MemoryLoader &&src) noexcept {
    this->mData = src.mData;
    this->mSize = src.mSize;
    src.mSize = 0;
    src.mData = nullptr;
}

MemoryLoader &MemoryLoader::operator=(
    athena::loaders::MemoryLoader &&src) noexcept {
    if (&src == this) {
        return *this;
    }

    this->mData = src.mData;
    this->mSize = src.mSize;
    src.mSize = 0;
    src.mData = nullptr;

    return *this;
}

void MemoryLoader::load(core::Allocator *allocator,
                        core::inner::Tensor *tensor) {
    auto pointer = reinterpret_cast<void *>(allocator->getRAMPointer(*tensor));
#ifdef DEBUG
    assert(pointer && "MemoryLoader pointer is NULL");
    assert(mSize <= tensor->getShapeView().getTotalSize() *
                        core::sizeOfDataType(tensor->getDataType()));
#endif
    std::memmove(pointer, mData, mSize);
}
std::string MemoryLoader::serialize() const {
    new core::FatalError(core::ATH_NOT_IMPLEMENTED, "Not serializable");
}

}  // namespace athena::loaders

namespace athena::core {
template <>
std::string
core::AbstractLoader::getLoaderName<athena::loaders::MemoryLoader>() {
    return "MemoryLoader";
}
}  // namespace athena::core

extern "C" {
void MemoryLoaderLoad(void *loader, void *allocator, void *tensor) {
    auto pLoader = reinterpret_cast<athena::loaders::MemoryLoader *>(loader);
    auto pAllocator = reinterpret_cast<athena::core::Allocator *>(allocator);
    auto pTensor = reinterpret_cast<athena::core::inner::Tensor *>(tensor);

#ifdef DEBUG
    assert(pLoader != nullptr);
    assert(pAllocator != nullptr);
    assert(pTensor != nullptr);
#endif
    pLoader->load(pAllocator, pTensor);
}

void *CreateMemoryLoader(void *data, size_t size) {
    return new athena::loaders::MemoryLoader(data, size);
}
}