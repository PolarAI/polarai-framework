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
#ifndef ATHENA_UTILS_H
#define ATHENA_UTILS_H

#include <llvm/TableGen/Record.h>

static std::string getMangledName(llvm::StringRef name, llvm::StringRef type) {
  std::string mangled = "athn_" + name.str();
  if (type == "float") {
    mangled += "_f";
  } else if (type == "double") {
    mangled += "_d";
  }
  return mangled;
}

static std::string getCStyleDeclaration(llvm::Record* record,
                                        llvm::StringRef gentype) {
  auto builtinName =
      record->getValue("builtinName")->getValue()->getAsUnquotedString();
  auto mangled = getMangledName(builtinName, gentype);

  std::string decl =
      "void " + mangled + "(void* devicePtr, void* allocatorPtr, ";
  auto args = record->getValueAsDag("arguments");
  for (size_t i = 0; i < args->arg_size(); i++) {
    auto argName = args->getArgName(i)->getAsUnquotedString();
    auto argType = args->getArg(i)->getAsUnquotedString();
    if (argType == "gentype") {
      decl += gentype.str() + " " + argName;
    } else if (argType == "tensor") {
      decl += "void* " + argName;
    }
    decl += ", ";
  }
  decl += "void* res)";
  return decl;
}

#endif // ATHENA_UTILS_H
