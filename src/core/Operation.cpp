/*
 * Copyright (c) 2018 Athena. All rights reserved.
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

#include <athena/core/Operation.h>

using namespace athena::core;
using namespace athena::core::inner;

template <enum DataType dataType>
struct DataTypeMapper;

template <>
struct DataTypeMapper<DataType::FLOAT> {
    using Type = float;
};

template <>
struct DataTypeMapper<DataType::DOUBLE> {
    using Type = double;
};

template <>
struct DataTypeMapper<DataType::HALF> {
    using Type = float;
};

template <enum DataType dataType>
struct FillByValue {
    template <typename ValueType>
    static void invoke(AbstractGenerator &g,
                       inner::Tensor &ownDerivative,
                       ValueType value) {
        typename DataTypeMapper<dataType>::Type internalValue = value;
        void *internalPointerToValue = reinterpret_cast<void *>(&internalValue);
        g.generate("fill", ownDerivative, internalPointerToValue);
    }
};

template <template <enum DataType, typename...> typename FunctionWrapper, typename ...Args>
static void DataTypeDispatchingCaller(DataType dataType, Args&& ...args) {
    switch (dataType) {
        case DataType::FLOAT:
            FunctionWrapper<DataType::FLOAT>::invoke(std::forward<Args>(args)...);
        case DataType::DOUBLE:
            FunctionWrapper<DataType::DOUBLE>::invoke(std::forward<Args>(args)...);
        case DataType::HALF:
            FunctionWrapper<DataType::HALF>::invoke(std::forward<Args>(args)...);
        case DataType::UNDEFINED:
            FatalError(FatalErrorType::ATH_BAD_TYPE, "Undefined type appears");
    }
}

namespace athena::core {
std::string Operation::getName() const {
    return mName;
}

void Operation::genOwnDerivative(AbstractGenerator& g,
                                 std::map<size_t, std::shared_ptr<inner::Tensor>> &outgoingDerivatives,
                                 inner::Tensor &ownDerivative) const {
    DataTypeDispatchingCaller<FillByValue>(ownDerivative.getDataType(), g, ownDerivative, 1.0);
    //for (auto& )
}
}  // namespace athena::core
