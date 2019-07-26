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

#ifndef ATHENA_FRAMEWORK_H
#define ATHENA_FRAMEWORK_H

#include <functional>
#include <string>
#include <typeindex>
#include <unordered_map>

class BaseTest;
using testmap = std::unordered_map<std::string, BaseTest*>;

class BaseTest {
    public:
    virtual void test() = 0;
    static testmap& registry();
};

template <typename T>
struct Registrar {
    explicit Registrar(std::string const& s) noexcept {
        BaseTest::registry()[s] = new T();
    }
};

#define STR(a) #a
#define XSTR(a) STR(a)

#define ATHENA_REGRESSION_TEST(TestCaseName)                     \
    class TestCaseName##Test : public BaseTest {                 \
        public:                                                  \
        static Registrar<TestCaseName##Test> registrar;          \
        void test() final;                                       \
    };                                                           \
    Registrar<TestCaseName##Test> TestCaseName##Test::registrar( \
        XSTR(__FILE__));                                         \
    void TestCaseName##Test::test()

#endif  // ATHENA_FRAMEWORK_H
