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

#include "WrapperBackend.h"
#include "utils.h"
#include <llvm/TableGen/TableGenBackend.h>

using namespace llvm;

void WrapperBackend::run(llvm::raw_ostream& o) {
  auto& defs = mRecords.getDefs();

  emitSourceFileHeader("Runtime wrappers", o);

  o << "#include <athena/backend/llvm/runtime/Device.h>\n";
  o << "#include <athena/backend/llvm/runtime/builtin.h>\n";
  o << "#include <athena/backend/llvm/BackendAllocator.h>\n";
  o << "#include <athena/core/inner/Tensor.h>\n";
  o << "using namespace athena::backend::llvm;\n";
  o << "using namespace athena::backend;\n";
  o << "using namespace athena::core::inner;\n";
  o << "using namespace athena::core;\n\n";


  o << "#include <cstring>\n\n";
  o << "extern \"C\" {\n";

  for (auto& d : defs) {
    auto superClasses = d.second->getSuperClasses();
    bool isSuitable = false;
    for (auto s : superClasses) {
      if (s.first->getName() == "RT_Builtin") {
        isSuitable = true;
      }
    }
    if (!isSuitable)
      continue;
    auto overloadDefs = d.second->getValueAsListOfDefs("overloadTypes");
    std::vector<std::string> overloads;
    overloads.reserve(overloadDefs.size());

    std::transform(overloadDefs.begin(), overloadDefs.end(),
                   std::back_inserter(overloads),
                   [&](Record* record) { return record->getName().str(); });
    for (const auto& overload : overloads) {
      o << "void " << getCStyleDeclaration(d.second.get(), overload) << " {\n";
      o << "auto* device = reinterpret_cast<Device*>(devicePtr);\n";
      o << "auto* allocator = "
           "reinterpret_cast<BackendAllocator*>(allocatorPtr);\n";
      std::string callArgs = "device, allocator";
      auto args = d.second->getValueAsDag("arguments");
      for (size_t i = 0; i < args->arg_size(); i++) {
        if (args->getArg(i)->getAsUnquotedString() == "tensor") {
          auto argName = args->getArgName(i)->getAsUnquotedString();
          o << "auto* " << argName << "Tensor = reinterpret_cast<Tensor*>("
            << argName << ");\n";
          callArgs += ", " + argName + "Tensor";
        }
      }
      // todo handle options
      o << d.second->getValue("builtinName")->getValue()->getAsUnquotedString()
        << "<" << overload << ">(" << callArgs << ");\n}\n";
    }
  }
  o << "}";
  /*

  o << "#include <athena/backend/llvm/runtime/Device.h>\n";
  o << "#include <athena/backend/llvm/runtime/builtin.h>\n";
  o << "#include <athena/backend/llvm/BackendAllocator.h>\n";
  o << "#include <athena/core/inner/Tensor.h>\n";
  o << "using namespace athena::backend::llvm;\n";
  o << "using namespace athena::backend;\n";
  o << "using namespace athena::core::inner;\n";
  o << "using namespace athena::core;\n\n";

  o << "#include <cstring>\n\n";
  o << "extern \"C\" {\n";

  for (auto& d : defs) {
    auto superClasses = d.second->getSuperClasses();
    bool isSuitable = false;
    for (auto s : superClasses) {
      if (s.first->getName() == "RT_Builtin") {
        isSuitable = true;
      }
    }
    if (!isSuitable)
      continue;
    auto overloadDefs = d.second->getValueAsListOfDefs("overloadTypes");
    std::vector<std::string> overloads;
    overloads.reserve(overloadDefs.size());

    std::transform(overloadDefs.begin(), overloadDefs.end(),
                   std::back_inserter(overloads),
                   [&](Record* record) { return record->getName().str(); });
    for (const auto& overload : overloads) {
      o << "void " << getCStyleDeclaration(d.second.get(), overload) << " {\n";
      o << "auto* device = reinterpret_cast<Device*>(devicePtr);\n";
      o << "auto* allocator = "
           "reinterpret_cast<BackendAllocator*>(allocatorPtr);\n";
      std::string callArgs = "device, allocator";
      auto args = d.second->getValueAsDag("arguments");
      for (auto arg : args->getArgNames()) {
        o << "auto* " << arg->getAsUnquotedString()
          << "Tensor = reinterpret_cast<Tensor*>(" << arg->getAsUnquotedString()
          << ");\n";
        callArgs += ", " + arg->getAsUnquotedString() + "Tensor";
      }
      // todo handle options
      o << d.second->getValue("builtinName")->getValue()->getAsUnquotedString()
        << "<" << overload << ">(" << callArgs << ");\n}\n";
    }
  }
  o << "}";*/
}
