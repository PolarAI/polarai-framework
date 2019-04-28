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

#ifndef ATHENA_LOADERFACTORY_H
#define ATHENA_LOADERFACTORY_H

#include <athena/core/AbstractLoader.h>
#include <athena/loaders/MemoryLoader/MemoryLoader.h>

#include <unordered_map>

namespace athena::core::inner {

class LoaderFactory {
    private:
    std::unordered_map<std::string, AbstractLoader *(*)(const std::string &)>
        loadersMap;

    LoaderFactory() {
        registerLoader<loaders::MemoryLoader>();
        registerLoader<DummyLoader>();
    }

    public:
    static LoaderFactory &getInstance() {
        static LoaderFactory loaderFactory;
        return loaderFactory;
    }

    static AbstractLoader *createLoader(const std::string &name,
                                        const std::string &data) {
        return getInstance().loadersMap[name](data);
    }
    template <typename T>
    void registerLoader() {
        loadersMap[T::template getLoaderName<T>()] = &T::deserialize;
    }
};
}  // namespace athena::core::inner

#endif  // ATHENA_LOADERFACTORY_H
