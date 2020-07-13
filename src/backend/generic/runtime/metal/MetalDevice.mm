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

#include "MetalDevice.hh"
#include "../utils/utils.hpp"
#include "MetalAllocator.hh"
#include "MetalEvent.hh"
#include "spirv_converter.hpp"

#include <polarai/backend/generic/BackendAllocator.hpp>
#include <polarai/backend/generic/runtime/Event.hpp>
#include <polarai/backend/generic/runtime/LaunchCommand.h>

namespace polarai::backend::generic {
MetalDevice::MetalDevice(id<MTLDevice> device) : mDevice(device) {
  mDeviceName = std::string([device.name UTF8String]);

  mAllocator = std::make_shared<MetalAllocator>(mDevice);
}

void MetalDevice::selectBinary(
    std::vector<std::shared_ptr<ProgramDesc>> &programs) {
  for (auto &prog : programs) {
    if (prog->type == ProgramDesc::Type::SPIRV_SHADER) {
      mProgram = prog;
      break;
    }
  }

  auto mslShader = convertSpvToMetal(mProgram->data);

  auto nsStrShader = [NSString stringWithCString:(mslShader.data())
                                        encoding:(NSASCIIStringEncoding)];
  mLibrary = [mDevice newLibraryWithSource:(nsStrShader)
                                   options:(nullptr)error:(nullptr)];
}

Event *MetalDevice::launch(BackendAllocator &allocator, LaunchCommand &cmd,
                           Event *blockingEvent) {
  if (blockingEvent) {
    blockingEvent->wait();
  }

  auto funcName = [NSString stringWithCString:(cmd.kernelName)
                                     encoding:(NSASCIIStringEncoding)];
  id<MTLFunction> function = [mLibrary newFunctionWithName:(funcName)];

  auto pipeline =
      [mDevice newComputePipelineStateWithFunction:(function) error:(nullptr)];

  auto commandQueue = [mDevice newCommandQueue];

  id<MTLCommandBuffer> commandBuffer = [commandQueue commandBuffer];
  id<MTLComputeCommandEncoder> computeEncoder =
      [commandBuffer computeCommandEncoder];

  [computeEncoder setComputePipelineState:(pipeline)];

  for (unsigned int i = 0; i < cmd.argsCount; i++) {
    if (cmd.args[i].type == ArgDesc::TENSOR) {
      auto tensor = static_cast<TensorInfo *>(cmd.args[i].arg);
      auto record = tensorInfoToRecord(tensor);
      id<MTLBuffer> *buf = allocator.get<id<MTLBuffer>>(record, *this);
      [computeEncoder setBuffer:(*buf) offset:(0)atIndex:(i)];
    } else {
      [computeEncoder setBytes:(cmd.args[i].arg)
                        length:(cmd.args[i].size)atIndex:(i)];
    }
  }

  auto &kernel = mProgram->kernels[cmd.kernelName];
  MTLSize gridSize =
      MTLSizeMake(kernel.globalX, kernel.globalY, kernel.globalZ);
  MTLSize threadgroupSize =
      MTLSizeMake(kernel.localX, kernel.localY, kernel.localZ);

  [computeEncoder dispatchThreads:gridSize
            threadsPerThreadgroup:threadgroupSize];

  [computeEncoder endEncoding];

  [commandBuffer commit];

  // todo figure out why it doesn't work from event
  [commandBuffer waitUntilCompleted];

  return new MetalEvent(this, commandBuffer);
}
}
