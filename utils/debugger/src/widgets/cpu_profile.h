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

#ifndef ATHENA_CPU_PROFILE_H
#define ATHENA_CPU_PROFILE_H

#include "imgui_widget_flamegraph.h"

#include <vector>

static void profilerValueGetter(float* startTimestamp, float* endTimestamp,
                                ImU8* level, const char** caption,
                                const void* data, int idx) {
  auto entry = *reinterpret_cast<const std::vector<ImVec2>*>(data);
  if (startTimestamp) {
    *startTimestamp = entry[idx].x;
  }
  if (endTimestamp) {
    *endTimestamp = entry[idx].y;
  }
  if (level) {
    *level = 3 - idx;
  }
  if (caption) {
    *caption = "Node X";
  }
}

void drawFlames() {
  std::vector<ImVec2> data = {ImVec2(0, 10), ImVec2(10, 15), ImVec2(20, 100)};
  ImGuiWidgetFlameGraph::PlotFlame("", &profilerValueGetter, &data, 3, 0,
                                   nullptr, FLT_MAX, FLT_MAX, ImVec2(1280, 0));
}

#endif // ATHENA_CPU_PROFILE_H
