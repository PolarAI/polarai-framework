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

#include <cuda.h>

#include <iostream>

inline void check(CUresult result) {
  if (result != CUDA_SUCCESS) {
    const char* errName;
    const char* errMsg;
    cuGetErrorName(result, &errName);
    cuGetErrorString(result, &errMsg);
    std::cerr << "CUDA Error " << errName << '\n' << errMsg << '\n';
    std::terminate();
  }
}
