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

#include "debugger.h"
#include "widgets/cpu_profile.h"
#include "widgets/graph.h"

void drawMainView(ImGuiIO& io, WindowState& state) {
  ImGuiWindowFlags window_flags =
      ImGuiWindowFlags_NoResize | ImGuiWindowFlags_NoCollapse;
  ImGui::SetNextWindowPos(ImVec2(0, 0));
  ImGui::SetNextWindowSize(ImVec2(io.DisplaySize.x, io.DisplaySize.y * 2 / 3));
  ImGui::Begin("Graph View", nullptr, window_flags);
  drawGraph();
  ImGui::End();

  ImGui::SetNextWindowPos(ImVec2(0, io.DisplaySize.y * 2 / 3));
  ImGui::SetNextWindowSize(ImVec2(io.DisplaySize.x, io.DisplaySize.y / 3));
  ImGui::Begin("Debugger", nullptr, window_flags);
  if (ImGui::BeginTabBar("##Tabs", ImGuiTabBarFlags_None)) {
    if (ImGui::BeginTabItem("Breakpoints")) {
      static bool checkbox1 = false;
      static bool checkbox2 = false;
      static bool checkbox3 = false;
      ImGui::Checkbox("Node A Before", &checkbox1);
      ImGui::Checkbox("Add Before", &checkbox2);
      ImGui::Checkbox("Add After", &checkbox3);
      ImGui::EndTabItem();
    }
    if (ImGui::BeginTabItem("Node Timing")) {
      drawFlames();
      ImGui::EndTabItem();
    }
    if (ImGui::BeginTabItem("CPU Load")) {
      ImGui::EndTabItem();
    }
    if (ImGui::BeginTabItem("RAM Consumption")) {
      ImGui::EndTabItem();
    }
    if (ImGui::BeginTabItem("MDAPI Statistics")) {
      ImGui::EndTabItem();
    }

    ImGui::EndTabBar();
  }
  ImGui::End();
}
