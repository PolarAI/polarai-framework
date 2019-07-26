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

#include "framework.h"

testmap &BaseTest::registry() {
    static testmap impl;
    return impl;
}

int main(int argc, char *argv[]) {
    if (argc != 2) return -1;

    std::string testName(argv[1]);

    auto registry = BaseTest::registry();

    auto *testObject = registry["\"" + testName + "\""];

    if (testObject) testObject->test();
}