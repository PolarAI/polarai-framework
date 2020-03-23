//===----------------------------------------------------------------------===//
// Copyright (c) 2020 Athena. All rights reserved.
// https://getathena.ml
//
// Licensed under MIT license.
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS, WITHOUT
// WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the
// License for the specific language governing permissions and limitations under
// the License.
//===----------------------------------------------------------------------===//

#pragma once

#include "Accessor.hpp"
#include "Index.hpp"

namespace picomath {

template <typename T, int Dims, access_mode Mode>
using global_accessor = Accessor<T, Dims, Mode>;

template <int Dims>
using id = Index<Dims>;

template <typename T, int Size>
using vec = T __attribute__((ext_vector_type(Size)));

} // namespace picomath