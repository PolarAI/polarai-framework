/*
 * Copyright (c) 2018 Athena. All rights reserved.
 * https://athenaframework.ml
 *
 * Licensed under MIT license.
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an “AS IS” BASIS, WITHOUT
 * WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the
 * License for the specific language governing permissions and limitations under
 * the License.
 */

#include <iostream>
#include <string>
#include <cassert>
#include "athena/backend/llvm/runtime-driver/runtime-driver.h"

std::string kPathToRuntimeCPU;
const std::string kPathToRuntimeCPUName = "PATH_TO_RUNTIME_CPU";

namespace athena::backend {
void test_creation() {
    std::string nameLibrary(kPathToRuntimeCPU);
    kRuntimeDriver.reload(nameLibrary);
    assert(kRuntimeDriver.isLoaded());
}

void test_loads() {
    std::string nameLibrary(kPathToRuntimeCPU);
    kRuntimeDriver = RuntimeDriver(nameLibrary);
    assert(kRuntimeDriver.isLoaded());
    kRuntimeDriver.unload();
    assert(!kRuntimeDriver.isLoaded());
    kRuntimeDriver.load(nameLibrary);
    assert(kRuntimeDriver.isLoaded());
    kRuntimeDriver.reload(nameLibrary);
    assert(kRuntimeDriver.isLoaded());
}

void test_using_functions() {
    std::string nameLibrary(kPathToRuntimeCPU);
    constexpr size_t size = 3;
    float vector_first[] = {1.0, 2.0, 3.0}, vector_second[] = {4.0, 5.0, 6.0},
        vector_res[size];
    fadd(vector_first, size, vector_second, size, vector_res);
    assert(vector_res[0] == 5.0);
    assert(vector_res[1] == 7.0);
    assert(vector_res[2] == 9.0);
}
}

void init() {
    kPathToRuntimeCPU = ::getenv(kPathToRuntimeCPUName.data());
}

int main() {
    init();
    ::athena::backend::test_creation();
    ::athena::backend::test_loads();
    ::athena::backend::test_using_functions();
    std::cout << "OK" << '\n';
    return 0;
}
