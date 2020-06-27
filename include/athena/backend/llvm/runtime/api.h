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

#ifndef ATHENA_API_H
#define ATHENA_API_H

#include <athena/backend/llvm/runtime/Context.h>

extern "C" {
athena::backend::llvm::Context* initContext();
void closeContext(athena::backend::llvm::Context*);
};

#endif // ATHENA_API_H
