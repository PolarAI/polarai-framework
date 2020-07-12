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

#if defined(_MSC_VER)
#define ATH_FORCE_INLINE __forceinline
#elif defined(__gcc__)
#define ATH_FORCE_INLINE inline __attribute__((always_inline))
#else
#define ATH_FORCE_INLINE inline
#endif

