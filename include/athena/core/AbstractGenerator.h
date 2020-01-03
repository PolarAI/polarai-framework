/*
 * Copyright (c) 2019 Athena. All rights reserved.
 * https://getathena.ml
 *
 * Licensed under MIT license.
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an “AS IS” BASIS, WITHOUT
 * WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the
 * License for the specific language governing permissions and limitations under
 * the License.
 */
#ifndef ATHENA_ABSTRACTGENERATOR_H
#define ATHENA_ABSTRACTGENERATOR_H

#include <athena/core/AbstractLoader.h>
#include <athena/core/core_export.h>
#include <athena/core/inner/Tensor.h>

namespace athena::core {

/**
 * A helper interface to connect operations with actual runtime implementations
 * through backend
 */
class ATH_CORE_EXPORT AbstractGenerator {
protected:
  virtual void generateImpl(std::string&, inner::Tensor& a) = 0;
  virtual void generateImpl(std::string&, inner::Tensor& a, void*& b) = 0;
  virtual void generateImpl(std::string&, inner::Tensor& a,
                            inner::Tensor& b) = 0;
  virtual void generateImpl(std::string&, inner::Tensor& a, inner::Tensor& b,
                            inner::Tensor& c) = 0;
  virtual void generateImpl(std::string&, inner::Tensor& a, uint64_t scaleA,
                            inner::Tensor& b, uint64_t scaleB,
                            inner::Tensor& c) = 0;
  virtual void generateImpl(std::string&, void*, inner::Tensor& a,
                            inner::Tensor& b, inner::Tensor& c) = 0;

public:
  virtual void openNode(std::string_view name) = 0;
  virtual void closeNode() = 0;

  /**
   * Creates empty function without arguments and sets it as current main
   * block
   * @param name Function name
   */
  virtual void generateFunctionHeader(const std::string& name) = 0;

  /**
   * Generates return command for current function and removes it from current
   * main block
   */
  virtual void generateFunctionFooter() = 0;

  /**
   * Generate code to execute loaders subroutines
   * @param loader Loader to be used
   * @param tensor Destination Tensor
   */
  virtual void generateLoad(const core::AbstractLoader& loader,
                            core::inner::Tensor& tensor) = 0;

  /**
   * Generate code that corresponds to given parameters
   * @tparam Args Arbitrary parameter pack defined by actual generator
   * implementation
   * @param name Name of operation/function/etc to be generated
   * @param a Parameters that are needed for code generation
   */
  template <typename... Args> void generate(std::string name, Args&&... a) {
    generateImpl(name, std::forward<Args>(a)...);
  };
};

} // namespace athena::core

#endif // ATHENA_ABSTRACTGENERATOR_H
