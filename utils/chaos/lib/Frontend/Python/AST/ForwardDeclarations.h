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

#ifndef ATHENA_FORWARDDECLARATIONS_H
#define ATHENA_FORWARDDECLARATIONS_H

#include <variant>

namespace chaos {
class TranslationUnitDecl;
class FunctionDecl;
class ParamDecl;
class BlockDecl;

class DeclStmt;
class PassStmt;

using AnyStatement = std::variant<DeclStmt, PassStmt>;
using AnyDecl = std::variant<FunctionDecl, ParamDecl, BlockDecl>;
} // namespace chaos

#endif // ATHENA_FORWARDDECLARATIONS_H
