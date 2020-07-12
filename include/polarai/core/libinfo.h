//===----------------------------------------------------------------------===//
// Copyright (c) 2020 PolarAI. All rights reserved.
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

#include <polar_core_export.h>

extern "C" {
POLAR_CORE_EXPORT const char* getMajorVersion();
POLAR_CORE_EXPORT const char* getMinorVersion();
POLAR_CORE_EXPORT const char* getPatchVersion();
POLAR_CORE_EXPORT const char* getBuildVersion();
}
