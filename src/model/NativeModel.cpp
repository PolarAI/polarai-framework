/*
 * Copyright (c) 2019 Athena. All rights reserved.
 * https://getathena.ml
 *
 * Licensed under MIT license.
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS, WITHOUT
 * WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the
 * License for the specific language governing permissions and limitations under
 * the License.
 */

#include <athena/core/DataType.h>
#include <athena/core/Graph.h>
#include <athena/core/InputNode.h>
#include <athena/core/LossNode.h>
#include <athena/core/Node.h>
#include <athena/core/OutputNode.h>
#include <athena/core/inner/LoaderFactory.h>
#include <athena/core/inner/OperationFactory.h>
#include <athena/model/NativeModel.h>

#include <Graph.pb.h>
#include <fstream>

using namespace athena::core;

namespace athena::model {
static proto_graph::Tensor_DataType nativeToProtoDataType(DataType type) {
    using Type = proto_graph::Tensor_DataType;
    switch (type) {
        case DataType::FLOAT:
            return Type::Tensor_DataType_FLOAT;
        case DataType::DOUBLE:
            return Type::Tensor_DataType_DOUBLE;
        case DataType::HALF:
            return Type::Tensor_DataType_HALF;
        default:
            return Type::Tensor_DataType_UNDEFINED;
    }
}
static proto_graph::Tensor serializeTensor(const inner::Tensor& tensor) {
    proto_graph::Tensor t;
    for (size_t i = 0; i < tensor.getShape().dimensions(); i++) {
        t.add_dimensions(tensor.getShape().dim(i));
    }
    t.set_data_type(nativeToProtoDataType(tensor.getDataType()));
    return t;
}
static proto_graph::Loader serializeLoader(const AbstractLoader& loader) {
    proto_graph::Loader l;
    l.set_name(loader.getName());
    l.set_data(loader.serialize());
    return l;
}
static proto_graph::Operation serializeOperation(const Operation& operation) {
    proto_graph::Operation op;
    op.set_name(operation.getName());
    op.set_data(operation.serialize());
    return op;
}
static void serializeNode(proto_graph::Node* actionNode, Node& node) {
    actionNode->set_index(node.getNodeIndex());
    actionNode->set_name(node.getName().data());
    auto* tensor = new proto_graph::Tensor(
        serializeTensor(inner::getTensorFromNode(node)));
    actionNode->set_allocated_tensor(tensor);
    auto* operation =
        new proto_graph::Operation(serializeOperation(node.getOperation()));
    actionNode->set_allocated_operation(operation);
    actionNode->set_inputs_count(node.getInputsCount());
}
static void serializeInputNode(proto_graph::InputNode* inputNode,
                               InputNode& node) {
    inputNode->set_index(node.getNodeIndex());
    inputNode->set_name(node.getName().data());
    auto* tensor = new proto_graph::Tensor(
        serializeTensor(inner::getTensorFromNode(node)));
    inputNode->set_allocated_tensor(tensor);
    auto* loader = new proto_graph::Loader(serializeLoader(node.getLoader()));
    inputNode->set_allocated_loader(loader);
    inputNode->set_is_frozen(node.isFrozen());
}
static void serializeLossNode(proto_graph::LossNode* lossNode, LossNode& node) {
    lossNode->set_index(node.getNodeIndex());
    lossNode->set_name(node.getName().data());
    auto* tensor = new proto_graph::Tensor(
        serializeTensor(inner::getTensorFromNode(node)));
    lossNode->set_allocated_tensor(tensor);
    auto* operation =
        new proto_graph::Operation(serializeOperation(node.getOperation()));
    lossNode->set_allocated_operation(operation);
    lossNode->set_criterion(static_cast<size_t>(node.getCriterion()));
    lossNode->set_inputs_count(node.getInputsCount());
}
static void serializeOutputNode(proto_graph::OutputNode* outputNode,
                                OutputNode& node) {
    outputNode->set_index(node.getNodeIndex());
    outputNode->set_name(node.getName().data());
    outputNode->set_inputs_count(node.getInputsCount());
}

void NativeModel::serializeGraph(core::Graph& graph, std::ostream& stream) {
    proto_graph::Graph savedGraph;

    auto topology = graph.getTopology();
    auto owningStorage = graph.getOwningStorage();
    auto syncStorage = graph.getSyncStorage();

    for (auto& edge : topology) {
        auto* e = savedGraph.add_edges();
        e->set_mark(edge.mark);
        e->set_end(edge.endNodeIndex);
        e->set_start(edge.startNodeIndex);
    }

    auto& inputNodes = std::get<std::vector<InputNode>>(owningStorage);
    for (auto& node : inputNodes) {
        serializeInputNode(savedGraph.add_input_nodes(), node);
    }
    auto& actionNodes = std::get<std::vector<Node>>(owningStorage);
    for (auto& node : actionNodes) {
        serializeNode(savedGraph.add_nodes(), node);
    }
    auto& lossNodes = std::get<std::vector<LossNode>>(owningStorage);
    for (auto& node : lossNodes) {
        serializeLossNode(savedGraph.add_loss_nodes(), node);
    }
    auto& outputNodes = std::get<std::vector<OutputNode>>(owningStorage);
    for (auto& node : outputNodes) {
        serializeOutputNode(savedGraph.add_output_nodes(), node);
    }

    auto& ctx = inner::getContext(graph);

    for (size_t idx : syncStorage) {
        auto* syncNode = inner::getNodeTable(ctx)[idx];
#ifdef DEBUG
        assert(syncNode);
#endif
        if (syncNode->getType() == NodeType::DEFAULT) {
            serializeNode(savedGraph.add_nodes(),
                          *node_dyncast<Node*>(syncNode));
        } else if (syncNode->getType() == NodeType::INPUT) {
            serializeInputNode(savedGraph.add_input_nodes(),
                               *node_dyncast<InputNode*>(syncNode));
        } else if (syncNode->getType() == NodeType::LOSS) {
            serializeLossNode(savedGraph.add_loss_nodes(),
                              *node_dyncast<LossNode*>(syncNode));
        } else if (syncNode->getType() == NodeType::OUTPUT) {
            serializeOutputNode(savedGraph.add_output_nodes(),
                                *node_dyncast<OutputNode*>(syncNode));
        } else {
            new FatalError(1, "Attempt to serialize unsupported node type: ",
                           syncNode->getName());
        }
    }

    savedGraph.SerializePartialToOstream(&stream);
}
void NativeModel::saveGraphToFile(core::Graph& graph,
                                  const std::string& filename) {
    std::ofstream out(filename);
    NativeModel::serializeGraph(graph, out);
    out.close();
}

static TensorShape getShape(const proto_graph::Tensor& t) {
    std::vector<size_t> dims(t.dimensions().begin(), t.dimensions().end());
    TensorShape shape(dims);
    return std::move(shape);
}
static DataType protoToNativeDataType(const proto_graph::Tensor& t) {
    using Type = proto_graph::Tensor_DataType;
    switch (t.data_type()) {
        case Type::Tensor_DataType_FLOAT:
            return DataType::FLOAT;
        case Type::Tensor_DataType_DOUBLE:
            return DataType::DOUBLE;
        case Type::Tensor_DataType_HALF:
            return DataType::HALF;
        default:
            return DataType::UNDEFINED;
    }
}
AbstractLoader* deserializeLoader(const proto_graph::Loader& loader) {
    // this looks like cause for memory leak, needs investigation
    return inner::LoaderFactory::createLoader(loader.name(), loader.data());
}

Operation* deserializeOperation(const proto_graph::Operation& operation) {
    // this looks like cause for memory leak, needs investigation
    return inner::OperationFactory::createOperation(operation.name(),
                                                    operation.data());
}

void NativeModel::deserializeGraph(core::Graph& graph, std::istream& stream) {
    proto_graph::Graph protograph;
    protograph.ParseFromIstream(&stream);

    std::unordered_map<size_t, size_t> nodesRemap;

    auto& ctx = inner::getContext(graph);

    for (auto& node : protograph.input_nodes()) {
        InputNode n(
            getShape(node.tensor()), protoToNativeDataType(node.tensor()),
                    *deserializeLoader(node.loader()), ctx, node.is_frozen(),
                    node.name());
        graph.addNode(n);
        nodesRemap[node.index()] = n.getNodeIndex();
    }

    for (auto& node : protograph.nodes()) {
        Node n(*deserializeOperation(node.operation()), ctx, node.name());
        graph.addNode(n);
        nodesRemap[node.index()] = n.getNodeIndex();
    }

    for (auto& node : protograph.loss_nodes()) {
        LossNode n(*deserializeOperation(node.operation()),
                   static_cast<Criterion>(node.criterion()), ctx, node.name());
        graph.addNode(n);
        nodesRemap[node.index()] = n.getNodeIndex();
    }

    for (auto& node : protograph.output_nodes()) {
        // OutputNode creates 0-dimension tensor, so UNDEFINED is ok here
        OutputNode n(DataType::UNDEFINED, ctx, node.name());
        graph.addNode(n);
        nodesRemap[node.index()] = n.getNodeIndex();
    }

    for (auto& edge : protograph.edges()) {
        auto* startNode = inner::getNodeTable(ctx)[nodesRemap[edge.start()]];
        auto* endNode = inner::getNodeTable(ctx)[nodesRemap[edge.end()]];
        endNode->after(*startNode, edge.mark());
    }
}
void NativeModel::readGraphFromFile(core::Graph& graph,
                                    const std::string& name) {
    std::ifstream stream(name);
    NativeModel::deserializeGraph(graph, stream);
    stream.close();
}
}