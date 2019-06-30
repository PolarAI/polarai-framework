/*
 * Copyright (c) 2018 Athena. All rights reserved.
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

#ifndef ATHENA_SETTINGS_H
#define ATHENA_SETTINGS_H

#if __has_cpp_attribute(clang::reinitializes)
#define ATHENA_REINITIALIZE [[clang::reinitializes]]
#else
#define ATHENA_REINITIALIZE
#endif

#endif //ATHENA_SETTINGS_H
