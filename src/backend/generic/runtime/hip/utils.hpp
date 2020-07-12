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

#pragma once

#include <hip/hip_runtime.h>

inline auto decodeHipError(hipError_t err) -> std::string {
  switch (err) {
  case hipSuccess:
    return "hipSuccess";
  case hipErrorInvalidContext:
    return "hipErrorInvalidContext";
  case hipErrorInvalidKernelFile:
    return "hipErrorInvalidKernelFile";
  case hipErrorMemoryAllocation:
    return "hipErrorMemoryAllocation";
  case hipErrorInitializationError:
    return "hipErrorInitializationError";
  case hipErrorLaunchFailure:
    return "hipErrorLaunchFailure";
  case hipErrorLaunchOutOfResources:
    return "hipErrorLaunchOutOfResources";
  case hipErrorInvalidDevice:
    return "hipErrorInvalidDevice";
  case hipErrorInvalidValue:
    return "hipErrorInvalidValue";
  case hipErrorInvalidDevicePointer:
    return "hipErrorInvalidDevicePointer";
  default:
    return "unknown";
  }
}
