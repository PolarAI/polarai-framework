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

#include "RuntimeInfoBackend.h"
#include <llvm/TableGen/TableGenBackend.h>

using namespace llvm;

void RuntimeInfoBackend::run(llvm::raw_ostream& o) {
  auto& defs = mRecords.getDefs();

  emitSourceFileHeader("Runtime information routines", o);
  o << "#include <cstring>\n\n";
  o << "extern \"C\" {\n";
  o << "  bool hasFeature(const char* featureName) {\n";
  for (auto& d : defs) {
    auto superClasses = d.second->getSuperClasses();
    bool isSuitable = false;
    for (auto s : superClasses) {
      if (s.first->getName() == "Runtime") {
        isSuitable = true;
      }
    }
    if (!isSuitable)
      continue;

    auto supportedTypes = d.second->getValueAsListOfDefs("supportedTypes");
    std::vector<std::string> types;
    types.reserve(supportedTypes.size());

    std::transform(supportedTypes.begin(), supportedTypes.end(),
                   std::back_inserter(types),
                   [&](Record* record) { return record->getName().str(); });
    for (auto& type : types) {
      o << "    if (strcmp(featureName, \"" + type + "\") == 0) { ";
      o << "return true; }\n";
    }

    auto features = d.second->getValueAsListOfStrings("extraFeatures");
    for (auto& feature : features) {
      o << "    if (strcmp(featureName, \"" + feature.str() + "\") == 0) { ";
      o << "return true; }\n";
    }
    break;
  }
  o << "    return false;\n  }\n";

  o << "  bool hasBuiltin(const char* builtinName) {\n";

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
    o << "    if (strcmp(builtinName, \"" +
             d.second->getValueAsString("builtinName").str() + "\") == 0) { ";
    o << "return true; }\n";
  }
  o << "    return false;\n  }\n";
  o << "}";
}
