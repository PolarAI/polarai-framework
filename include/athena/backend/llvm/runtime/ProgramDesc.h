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

#ifndef ATHENA_PROGRAMDESC_H
#define ATHENA_PROGRAMDESC_H

struct ProgramDesc {
  enum ProgramType { SPIRV, BINARY, TEXT };

  ProgramType type;
  size_t length;
  const char* data;
};

#endif // ATHENA_PROGRAMDESC_H
