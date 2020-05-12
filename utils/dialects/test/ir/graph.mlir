// RUN: athena-opt %s
module {
  "ath_graph.node"() ( {
  ^bb0(%arg0: index, %arg1: index):	// no predecessors
    %0 = "ath_graph.get_tensor"(%arg0) {virtual_address = 1 : index} : (index) -> tensor<1x8xf32>
    %1 = "ath_graph.slice"(%arg1, %0) : (index, tensor<1x8xf32>) -> tensor<8xf32>
    "ath_graph.alloc"(%1) : (tensor<8xf32>) -> ()
    "ath_graph.lock"(%1) {lock_type = "read_write"} : (tensor<8xf32>) -> ()
    "ath_graph.invoke_loader"(%1) {loader_routine = "MyLoaderLoad"} : (tensor<8xf32>) -> ()
    "ath_graph.release"(%1) : (tensor<8xf32>) -> ()
    ath_graph.return %0 : tensor<1x8xf32>
  }) {cluster_id = 0 : index, node_id = 0 : index, sym_name = "InputA", type = (index, index) -> tensor<1x8xf32>} : () -> ()
  "ath_graph.node"() ( {
  ^bb0(%arg0: index, %arg1: index):	// no predecessors
    %0 = "ath_graph.get_tensor"(%arg0) {virtual_address = 9 : index} : (index) -> tensor<1x8xf32>
    %1 = "ath_graph.slice"(%arg1, %0) : (index, tensor<1x8xf32>) -> tensor<8xf32>
    "ath_graph.alloc"(%1) : (tensor<8xf32>) -> ()
    "ath_graph.lock"(%1) {lock_type = "read_write"} : (tensor<8xf32>) -> ()
    "ath_graph.invoke_loader"(%1) {loader_routine = "MyLoaderLoad"} : (tensor<8xf32>) -> ()
    "ath_graph.release"(%1) : (tensor<8xf32>) -> ()
    ath_graph.return %0 : tensor<1x8xf32>
  }) {cluster_id = 0 : index, node_id = 1 : index, sym_name = "InputB", type = (index, index) -> tensor<8xf32>} : () -> ()
  "ath_graph.node"() ( {
  ^bb0(%arg0: tensor<1x8xf32>, %arg1: tensor<1x8xf32>, %arg2: index, %arg3: index):	// no predecessors
    %0 = "ath_graph.get_tensor"(%arg2) {virtual_address = 17 : index} : (index) -> tensor<1x8xf32>
    %1 = "ath_graph.slice"(%arg3, %0) : (index, tensor<1x8xf32>) -> tensor<8xf32>
    %2 = "ath_graph.slice"(%arg3, %arg0) : (index, tensor<1x8xf32>) -> tensor<8xf32>
    %3 = "ath_graph.slice"(%arg3, %arg1) : (index, tensor<1x8xf32>) -> tensor<8xf32>
    "ath_graph.alloc"(%1) : (tensor<8xf32>) -> ()
    "ath_graph.lock"(%2) {lock_type = "read"} : (tensor<8xf32>) -> ()
    "ath_graph.lock"(%3) {lock_type = "read"} : (tensor<8xf32>) -> ()
    "ath_graph.lock"(%1) {lock_type = "read_write"} : (tensor<8xf32>) -> ()
    %cst = constant 1.000000e+00 : f32
    %4 = "ath_graph.add"(%2, %cst, %3, %cst, %1) : (tensor<8xf32>, f32, tensor<8xf32>, f32, tensor<8xf32>) -> tensor<8xf32>
    "ath_graph.release"(%1) : (tensor<8xf32>) -> ()
    "ath_graph.release"(%2) : (tensor<8xf32>) -> ()
    "ath_graph.release"(%3) : (tensor<8xf32>) -> ()
    ath_graph.return %0 : tensor<1x8xf32>
  }) {cluster_id = 1 : index, node_id = 2 : index, sym_name = "SumNode", type = (tensor<1x8xf32>, tensor<1x8xf32>, index, index) -> tensor<1x8xf32>} : () -> ()
  "ath_graph.graph"() ( {
  ^bb0(%arg0: index, %arg1: index):	// no predecessors
    %0 = ath_graph.eval @InputA(%arg0, %arg1) : (index, index) -> tensor<1x8xf32>
    %1 = ath_graph.eval @InputB(%arg0, %arg1) : (index, index) -> tensor<1x8xf32>
    %2 = ath_graph.eval @SumNode(%0, %1, %arg0, %arg1) : (tensor<1x8xf32>, tensor<1x8xf32>, index, index) -> tensor<1x8xf32>
    "ath_graph.graph_terminator"() : () -> ()
  }) {sym_name = "SampleGraph", type = (index, index) -> ()} : () -> ()
}
