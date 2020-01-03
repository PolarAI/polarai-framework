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

#include <athena/core/libinfo.h>

const char* getMajorVersion() { return ATHENA_MAJOR_VERSION; }

const char* getMinorVersion() { return ATHENA_MINOR_VERSION; }

const char* getPatchVersion() { return ATHENA_PATCH_VERSION; }

const char* getBuildVersion() {
#ifdef ATHENA_CI_BUILD_NUMBER
  return ATHENA_CI_BUILD_NUMBER;
#else
  return "dev";
#endif
}