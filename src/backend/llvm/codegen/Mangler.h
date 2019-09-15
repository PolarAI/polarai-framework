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

#ifndef ATHENA_MANGLER_H
#define ATHENA_MANGLER_H

#include <string>

namespace athena::backend::llvm {
class Mangler {
    private:
    template <typename T>
    static std::string getTypePostfix();

    public:
    template <typename T>
    static std::string getMangledName(const std::string &name) {
        return "athn_" + name + getTypePostfix<T>();
    }
};
}  // namespace athena::backend::llvm

#endif  // ATHENA_MANGLER_H
