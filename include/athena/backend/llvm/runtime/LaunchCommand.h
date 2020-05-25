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

// fixme this file should be autogenerated.

#ifndef ATHENA_LAUNCHCOMMAND_H
#define ATHENA_LAUNCHCOMMAND_H

struct ArgDesc {
  enum ArgType { TENSOR = 0, DATA = 1 };
  uint64_t size;
  void* arg;
  ArgType type;
};

struct LaunchCommand {
  const char* kernelName;
  uint64_t argsCount;
  ArgDesc* args;
  uint64_t workDim;
  uint64_t* globalSize;
  uint64_t* localSize;
};

#endif // ATHENA_LAUNCHCOMMAND_H
