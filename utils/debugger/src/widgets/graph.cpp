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

#include "graph.h"

#include "imgui.h"
#include "imnodes.h"

void drawGraph() {
  imnodes::BeginNodeEditor();

  imnodes::BeginNode(1);

  imnodes::BeginNodeTitleBar();
  ImGui::TextUnformatted("Input A");
  imnodes::EndNodeTitleBar();

  imnodes::BeginOutputAttribute(2);
  ImGui::Indent(40);
  ImGui::Text("output");
  imnodes::EndAttribute();
  imnodes::EndNode();

  imnodes::BeginNode(3);

  imnodes::BeginNodeTitleBar();
  ImGui::TextUnformatted("Input B");
  imnodes::EndNodeTitleBar();

  imnodes::BeginOutputAttribute(4);
  ImGui::Indent(40);
  ImGui::Text("output");
  imnodes::EndAttribute();
  imnodes::EndNode();

  imnodes::BeginNode(5);

  imnodes::BeginNodeTitleBar();
  ImGui::TextUnformatted("Add");
  imnodes::EndNodeTitleBar();

  imnodes::BeginInputAttribute(6);
  ImGui::Text("a");
  imnodes::EndAttribute();
  imnodes::BeginInputAttribute(7);
  ImGui::Text("b");
  imnodes::EndAttribute();
  imnodes::EndNode();

  imnodes::Link(8, 2, 6);
  imnodes::Link(9, 4, 7);

  imnodes::EndNodeEditor();
}
