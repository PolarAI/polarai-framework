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

#include <polar_core_export.h>

#include <string>

#ifndef ATHENA_LIBINFO_H
#define ATHENA_LIBINFO_H

extern "C" {
POLAR_CORE_EXPORT const char* getMajorVersion();
POLAR_CORE_EXPORT const char* getMinorVersion();
POLAR_CORE_EXPORT const char* getPatchVersion();
POLAR_CORE_EXPORT const char* getBuildVersion();
}

#endif // ATHENA_LIBINFO_H
