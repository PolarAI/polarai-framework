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

#include <athena/utils/error/FatalError.h>
#include <athena/utils/string/String.h>
#include <athena/utils/string/StringView.h>

namespace athena::utils {
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
} // namespace athena::utils
