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

#ifndef ATHENA_DEFINES_H
#define ATHENA_DEFINES_H

#if defined(_MSC_VER)
#define ATH_FORCE_INLINE __forceinline
#elif defined(__gcc__)
#define ATH_FORCE_INLINE inline __attribute__((always_inline))
#else
#define ATH_FORCE_INLINE inline
#endif

#endif // ATHENA_DEFINES_H
