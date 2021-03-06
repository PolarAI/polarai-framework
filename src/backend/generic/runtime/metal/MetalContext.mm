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

#include "MetalContext.hh"
#include "MetalDevice.hh"

#import <Metal/Metal.h>

namespace polarai::backend::generic {
MetalContext::MetalContext() {
  NSArray<id<MTLDevice>> *deviceList = MTLCopyAllDevices();

  for (NSUInteger i = 0; i < [deviceList count]; i++) {
    mDevices.push_back(std::make_shared<MetalDevice>(deviceList[i]));
  }
}
}
