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

#ifndef ATHENA_STRUCTS_H
#define ATHENA_STRUCTS_H

namespace athena::backend {
template <typename T> struct GEMMOptions {
  bool transposeA;
  bool transposeB;
  T alpha;
  T beta;
};

template <typename T> struct HadamardOptions {
  T alpha;
  T beta;
};
} // namespace athena::backend

#endif // ATHENA_STRUCTS_H
