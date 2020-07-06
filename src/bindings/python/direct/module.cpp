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

#include "Context.hpp"
#include "Graph.hpp"
#include "Node.hpp"

#include <pybind11/pybind11.h>

namespace py = pybind11;
namespace p = polar::python;

PYBIND11_MODULE(_polar_direct, m) {
  py::class_<p::PyGraph>(m, "Graph")
      .def(py::init<p::PyContext, const std::string&>())
      .def("get_name", &p::PyGraph::getName);
  py::class_<p::PyAbstractNode>(m, "AbstractNode");
  py::class_<p::PyNode, p::PyAbstractNode>(m, "Node");
  py::class_<p::PyContext>(m, "Context").def(py::init<>());
}
