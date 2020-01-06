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

#include "MarkdownBackend.h"
#include "utils.h"
#include <iostream>
#include <numeric>

using namespace llvm;

void MarkdownBackend::run(raw_ostream& o) {
  // emit header
  o << "<!-- This file is generated automatically with athena-tablegen. Do not "
       "edit manually. -->\n\n";
  o << "# Runtime builtins reference\n\n";
  auto& defs = mRecords.getDefs();
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

    o << "## "
      << d.second->getValue("builtinName")->getValue()->getAsUnquotedString()
      << "\n\n";
    o << "> "
      << d.second->getValue("summary")->getValue()->getAsUnquotedString()
      << "\n\n";

    auto overloadDefs = d.second->getValueAsListOfDefs("overloadTypes");
    std::vector<std::string> overloads;
    overloads.reserve(overloadDefs.size());

    std::transform(overloadDefs.begin(), overloadDefs.end(),
                   std::back_inserter(overloads),
                   [&](Record* record) { return record->getName().str(); });

    o << "**Available overloads:** "
      << std::accumulate(overloads.begin(), overloads.end(), std::string(),
                         [](std::string& ss, std::string& s) {
                           return ss.empty() ? s : ss + ", " + s;
                         })
      << "\n\n";

    o << "**C-style declarations**:\n```\n";
    for (const auto& overload : overloads) {
      o << getCStyleDeclaration(d.second.get(), overload) << ";\n";
    }

    o << "```\n";

    o << "### Detailed description\n\n";
    o << d.second->getValue("description")->getValue()->getAsUnquotedString();
  }
}
