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

#include "spirv_converter.hpp"

std::string convertSpvToMetal(std::vector<char>& module) {
  auto reintPtr = reinterpret_cast<uint32_t*>(module.data());
  spirv_cross::CompilerMSL compiler(reintPtr, module.size() / sizeof(uint32_t));

  spirv_cross::CompilerMSL::Options options;
  options.set_msl_version(2, 3, 0);
  options.platform = spirv_cross::CompilerMSL::Options::macOS;

  return compiler.compile();
}