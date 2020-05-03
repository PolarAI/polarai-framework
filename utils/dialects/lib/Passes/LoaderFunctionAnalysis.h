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

#ifndef ATHENA_LOADERFUNCTIONANALYSIS_H
#define ATHENA_LOADERFUNCTIONANALYSIS_H

#include "mlir/IR/Module.h"
#include <bits/unordered_set.h>

/// Gathers loader function names to deploy function definitions later.
class LoaderFunctionAnalysis {
  std::unordered_set<std::string> mLoaders;

public:
  LoaderFunctionAnalysis(mlir::Operation* op);
  const std::unordered_set<std::string>& getLoaderFunctionNames() const {
    return mLoaders;
  }
};

#endif // ATHENA_LOADERFUNCTIONANALYSIS_H
