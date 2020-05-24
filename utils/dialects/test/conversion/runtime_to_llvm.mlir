
module {
  func @inputA(%arg0: !ath_rt.graph_handle) -> !ath_rt.event attributes {cluster_id = 0 : index, node_id = 0 : index} {
    %0 = "ath_graph.create_tensor"() {virtual_address = 1 : index} : () -> tensor<8xf32>
    "ath_graph.alloc"(%0) : (tensor<8xf32>) -> ()
    "ath_graph.lock"(%0) {lock_type = "read_write"} : (tensor<8xf32>) -> ()
    "ath_graph.invoke_loader"(%0) : (tensor<8xf32>) -> ()
    "ath_graph.release"(%0) : (tensor<8xf32>) -> ()
    %1 = "ath_rt.null_event"() : () -> !ath_rt.event
    return %1 : !ath_rt.event
  }
  func @inputB(%arg0: !ath_rt.graph_handle) -> !ath_rt.event attributes {cluster_id = 0 : index, node_id = 1 : index} {
    %0 = "ath_graph.create_tensor"() {virtual_address = 9 : index} : () -> tensor<8xf32>
    "ath_graph.alloc"(%0) : (tensor<8xf32>) -> ()
    "ath_graph.lock"(%0) {lock_type = "read_write"} : (tensor<8xf32>) -> ()
    "ath_graph.invoke_loader"(%0) : (tensor<8xf32>) -> ()
    "ath_graph.release"(%0) : (tensor<8xf32>) -> ()
    %1 = "ath_rt.null_event"() : () -> !ath_rt.event
    return %1 : !ath_rt.event
  }
  func @sum(%arg0: !ath_rt.graph_handle) -> !ath_rt.event attributes {cluster_id = 1 : index, node_id = 2 : index} {
    %0 = "ath_graph.create_tensor"() {virtual_address = 9 : index} : () -> tensor<8xf32>
    %1 = "ath_graph.create_tensor"() {virtual_address = 1 : index} : () -> tensor<8xf32>
    %2 = "ath_graph.create_tensor"() {virtual_address = 17 : index} : () -> tensor<8xf32>
    "ath_graph.lock"(%1) {lock_type = "read"} : (tensor<8xf32>) -> ()
    "ath_graph.lock"(%0) {lock_type = "read"} : (tensor<8xf32>) -> ()
    "ath_graph.alloc"(%2) : (tensor<8xf32>) -> ()
    "ath_graph.lock"(%2) {lock_type = "read_write"} : (tensor<8xf32>) -> ()
    %cst = constant 1.000000e+00 : f32
    %3 = "ath_rt.select_device"() {nodeId = 2 : index} : () -> !ath_rt.device
    %4 = "ath_rt.launch"(%3, %1, %cst, %0, %cst, %2) {kernel = ""} : (!ath_rt.device, tensor<8xf32>, f32, tensor<8xf32>, f32, tensor<8xf32>) -> !ath_rt.event
    "ath_graph.release"(%1) : (tensor<8xf32>) -> ()
    "ath_graph.release"(%0) : (tensor<8xf32>) -> ()
    "ath_graph.release"(%2) : (tensor<8xf32>) -> ()
    return %4 : !ath_rt.event
  }
  func @mainGraph(%arg0: !ath_rt.graph_handle) {
    %0 = call @inputA(%arg0) : (!ath_rt.graph_handle) -> !ath_rt.event
    %1 = call @inputB(%arg0) : (!ath_rt.graph_handle) -> !ath_rt.event
    "ath_rt.barrier"(%0, %1) : (!ath_rt.event, !ath_rt.event) -> ()
    %2 = call @sum(%arg0) : (!ath_rt.graph_handle) -> !ath_rt.event
    return
  }
}
