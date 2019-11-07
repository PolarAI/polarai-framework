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

#include <athena/core/FatalError.h>

#include <gtest/gtest.h>

using namespace athena::core;

TEST(FatalError, FatalErrorAbortsExecution) {
#ifdef DEBUG
    ASSERT_DEBUG_DEATH(new FatalError(ATH_FATAL_OTHER, "FatalError test"),
                       "FatalError test");
#else
    EXPECT_EXIT(new FatalError(ATH_FATAL_OTHER, "FatalError test"),
                ::testing::ExitedWithCode(ATH_FATAL_OTHER), "FatalError test");
#endif
}