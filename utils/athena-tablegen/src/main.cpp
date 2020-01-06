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

#include "../lib/MarkdownBackend.h"
#include "../lib/RuntimeInfoBackend.h"
#include "../lib/WrapperBackend.h"
#include <llvm/Support/CommandLine.h>
#include <llvm/TableGen/Main.h>

using namespace llvm;

bool emitMarkdownDocs(raw_ostream& OS, RecordKeeper& Records) {
  MarkdownBackend(Records).run(OS);
  return false;
}

bool emitRTInfo(raw_ostream& OS, RecordKeeper& Records) {
  RuntimeInfoBackend(Records).run(OS);
  return false;
}

bool emitWrapper(raw_ostream& OS, RecordKeeper& Records) {
  WrapperBackend(Records).run(OS);
  return false;
}

int main(int argc, char** argv) {
  cl::opt<bool> EmitDocs(
      "emit-markdown-docs",
      cl::desc("Emit builtin documentation in Markdown format"));
  cl::opt<bool> EmitRTInfo("emit-runtime-info",
                           cl::desc("Emit runtime information routines"));
  cl::opt<bool> EmitWrapper("emit-wrapper",
                            cl::desc("Emit runtime C-style wrapper"));
  cl::ParseCommandLineOptions(argc, argv);

  if (EmitDocs.getValue()) {
    return TableGenMain(argv[0], &emitMarkdownDocs);
  } else if (EmitRTInfo.getValue()) {
    return TableGenMain(argv[0], &emitRTInfo);
  } else if (EmitWrapper) {
    return TableGenMain(argv[0], &emitWrapper);
  }

  return 0;
}
