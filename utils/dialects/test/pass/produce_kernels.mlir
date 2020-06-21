module {
  "compute.module"() ( {
    compute.module_end
  }) {sym_name = "kernels"} : () -> ()
  "ath_graph.node"() ( {
    %0 = "ath_graph.create_tensor"() {virtual_address = 9 : index} : () -> tensor<8xf32>
    %1 = "ath_graph.create_tensor"() {virtual_address = 1 : index} : () -> tensor<8xf32>
    %2 = "ath_graph.create_tensor"() {virtual_address = 17 : index} : () -> tensor<8xf32>
    "ath_graph.lock"(%1) {lock_type = "read"} : (tensor<8xf32>) -> ()
    "ath_graph.lock"(%0) {lock_type = "read"} : (tensor<8xf32>) -> ()
    "ath_graph.alloc"(%2) : (tensor<8xf32>) -> ()
    "ath_graph.lock"(%2) {lock_type = "read_write"} : (tensor<8xf32>) -> ()
    %cst = constant 1.000000e+00 : f32
    %cst1 = constant 8 : index
    %3 = "ath_graph.add"(%1, %cst, %0, %cst, %cst1, %2) : (tensor<8xf32>, f32, tensor<8xf32>, f32, index, tensor<8xf32>) -> tensor<8xf32>
    "ath_graph.release"(%1) : (tensor<8xf32>) -> ()
    "ath_graph.release"(%0) : (tensor<8xf32>) -> ()
    "ath_graph.release"(%2) : (tensor<8xf32>) -> ()
    ath_graph.return %3 : tensor<8xf32>
  }) {cluster_id = 1 : index, node_id = 2 : index, sym_name = "sum", type = () -> tensor<8xf32>} : () -> ()
}
