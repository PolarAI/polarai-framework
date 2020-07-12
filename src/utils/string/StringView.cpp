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

#include <polarai/utils/error/FatalError.hpp>
#include <polarai/utils/string/String.hpp>
#include <polarai/utils/string/StringView.hpp>

namespace polarai::utils {
StringView::StringView(const String& string) : mString(&string) {}

const char* StringView::getString() const {
#ifdef DEBUG
  if (!mString) {
    FatalError(ATH_BAD_ACCESS, "String ", this,
               " getting isn't completed. String pointer is nullptr.");
  }
#endif
  return mString->getString();
}

size_t StringView::getSize() const {
#ifdef DEBUG
  if (!mString) {
    FatalError(ATH_BAD_ACCESS, "String ", this,
               " getting size isn't completed. String pointer is nullptr.");
  }
#endif
  return mString->getSize();
}
} // namespace polarai::utils
