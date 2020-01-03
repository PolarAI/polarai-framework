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
#include <athena/core/log.h>

namespace athena {

const std::unique_ptr<LogHolder> logHolder = std::make_unique<LogHolder>();

core::AbstractLogger& log() { return *(logHolder->mLog); }

core::AbstractLogger& err() { return *(logHolder->mErr); }
} // namespace athena