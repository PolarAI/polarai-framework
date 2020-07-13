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

#include <polarai/loaders/ConstantLoader.hpp>

using namespace polarai::loaders;

ConstantLoader::ConstantLoader(utils::SharedPtr<core::internal::ContextInternal> context,
  utils::Index publicIndex) : core::PublicEntity(std::move(context), publicIndex) {}

const internal::ConstantLoaderInternal* ConstantLoader::internal() const {
  return mContext->getPtr<internal::ConstantLoaderInternal>(mPublicIndex);
}

internal::ConstantLoaderInternal* ConstantLoader::internal() {
  return mContext->getPtr<internal::ConstantLoaderInternal>(mPublicIndex);
}

void ConstantLoader::setConstant(float constant) {
  internal()->setConstant(constant);
}
