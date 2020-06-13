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

#ifndef COMPUTE_DIALECT
#define COMPUTE_DIALECT

include "mlir/IR/OpBase.td"

def Compute_Dialect : Dialect {
    let name = "compute";
    let cppNamespace = "compute";
}

class Compute_Op<string mnemonic, list <OpTrait> traits = []> 
        : Op<Compute_Dialect, mnemonic, traits>;

#endif // COMPUTE_DIALECT
