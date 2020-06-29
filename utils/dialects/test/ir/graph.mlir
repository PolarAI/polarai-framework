// RUN: polar-opt %s

module {
  "polar_graph.node"() ( {
    %0 = "polar_graph.create_tensor"() {virtual_address = 1 : index} : () -> tensor<2x2xf32>
    "polar_graph.alloc"(%0) : (tensor<2x2xf32>) -> ()
    "polar_graph.lock"(%0) {lock_type = "read_write"} : (tensor<2x2xf32>) -> ()
    "polar_graph.invoke_loader"(%0) : (tensor<2x2xf32>) -> ()
    "polar_graph.release"(%0) : (tensor<2x2xf32>) -> ()
    polar_graph.return %0 : tensor<2x2xf32>
  }) {cluster_id = 0 : index, node_id = 4 : index, sym_name = "inp1", type = () -> tensor<2x2xf32>} : () -> ()
  "polar_graph.node"() ( {
    %0 = "polar_graph.create_tensor"() {virtual_address = 17 : index} : () -> tensor<2x2xf32>
    "polar_graph.alloc"(%0) : (tensor<2x2xf32>) -> ()
    "polar_graph.lock"(%0) {lock_type = "read_write"} : (tensor<2x2xf32>) -> ()
    "polar_graph.invoke_loader"(%0) : (tensor<2x2xf32>) -> ()
    "polar_graph.release"(%0) : (tensor<2x2xf32>) -> ()
    polar_graph.return %0 : tensor<2x2xf32>
  }) {cluster_id = 0 : index, node_id = 6 : index, sym_name = "inp2", type = () -> tensor<2x2xf32>} : () -> ()
  "polar_graph.node"() ( {
  ^bb0(%arg0: tensor<2x2xf32>, %arg1: tensor<2x2xf32>):  // no predecessors
    %0 = "polar_graph.create_tensor"() {virtual_address = 33 : index} : () -> tensor<2x2xf32>
    "polar_graph.alloc"(%0) : (tensor<2x2xf32>) -> ()
    %cst = constant 1.000000e+00 : f32
    %cst_0 = constant 1.000000e+00 : f32
    "polar_graph.lock"(%0) {lock_type = "read_write"} : (tensor<2x2xf32>) -> ()
    "polar_graph.lock"(%arg1) {lock_type = "read"} : (tensor<2x2xf32>) -> ()
    "polar_graph.lock"(%arg0) {lock_type = "read"} : (tensor<2x2xf32>) -> ()
    "polar_graph.add"(%arg0, %cst, %arg1, %cst_0, %0) : (tensor<2x2xf32>, f32, tensor<2x2xf32>, f32, tensor<2x2xf32>) -> ()
    "polar_graph.release"(%arg1) : (tensor<2x2xf32>) -> ()
    "polar_graph.release"(%arg0) : (tensor<2x2xf32>) -> ()
    "polar_graph.release"(%0) : (tensor<2x2xf32>) -> ()
    polar_graph.return %0 : tensor<2x2xf32>
  }) {cluster_id = 1 : index, node_id = 9 : index, sym_name = "add", type = (tensor<2x2xf32>, tensor<2x2xf32>) -> tensor<2x2xf32>} : () -> ()
  "polar_graph.graph"() ( {
    %0 = polar_graph.eval @inp1() : () -> tensor<2x2xf32>
    %1 = polar_graph.eval @inp2() : () -> tensor<2x2xf32>
    "polar_graph.barrier"() {clusterId = 0 : index} : () -> ()
    %2 = polar_graph.eval @add(%0, %1) : (tensor<2x2xf32>, tensor<2x2xf32>) -> tensor<2x2xf32>
    "polar_graph.barrier"() {clusterId = 1 : index} : () -> ()
    "polar_graph.barrier"() {clusterId = 2 : index} : () -> ()
    "polar_graph.graph_terminator"() : () -> ()
  }) {sym_name = "mainGraph", type = () -> ()} : () -> ()
}
