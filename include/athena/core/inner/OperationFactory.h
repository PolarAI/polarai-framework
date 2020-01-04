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

#ifndef ATHENA_OPERATIONFACTORY_H
#define ATHENA_OPERATIONFACTORY_H

#include <athena/core/Operation.h>
#include <athena/core/core_export.h>
#include <athena/ops/AddOperation.h>
#include <athena/ops/GEMMOperation.h>
#include <athena/ops/MSELossFunction.h>

#include <unordered_map>

namespace athena::core::inner {

class ATH_CORE_EXPORT OperationFactory {
    private:
    std::unordered_map<std::string, Operation *(*)(const std::string &)> opsMap;

    OperationFactory() {
        registerOperation<ops::AddOperation>("add");
        registerOperation<ops::MSELossFunction>("mse");
        registerOperation<ops::GEMMOperation>("gemm");
    }

    public:
    static OperationFactory &getInstance() {
        static OperationFactory operationFactory;
        return operationFactory;
    }

    static Operation *createOperation(const std::string &name,
                                      const std::string &data) {
        return getInstance().opsMap[name](data);
    }

    template <typename T>
    void registerOperation(const std::string &name) {
        opsMap[name] = &T::deserialize;
    }
};
}  // namespace athena::core::inner

#endif  // ATHENA_OPERATIONFACTORY_H
