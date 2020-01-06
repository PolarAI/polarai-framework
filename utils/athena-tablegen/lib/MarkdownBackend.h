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
#ifndef ATHENA_MARKDOWNBACKEND_H
#define ATHENA_MARKDOWNBACKEND_H

#include <llvm/Support/raw_ostream.h>
#include <llvm/TableGen/Main.h>
#include <llvm/TableGen/Record.h>
#include <llvm/TableGen/TableGenBackend.h>

class MarkdownBackend {
private:
  llvm::RecordKeeper& mRecords;

public:
  MarkdownBackend(llvm::RecordKeeper& R) : mRecords(R){};
  void run(llvm::raw_ostream& o);
};

#endif // ATHENA_MARKDOWNBACKEND_H
