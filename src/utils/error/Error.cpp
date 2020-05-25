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

#include <athena/utils/error/Error.h>

namespace athena::utils {
Error::Error() : mErrorCode(0) {}

Error::operator bool() const { return mErrorCode != 0; }

const String& Error::getMessage() const { return mErrorMessage; }
} // namespace athena::utils
