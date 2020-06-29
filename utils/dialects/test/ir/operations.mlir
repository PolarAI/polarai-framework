// RUN: polar-opt %s

module {
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
  }) {cluster_id = 1 : index, node_id = 9 : index, sym_name = "testAdd", type = (tensor<2x2xf32>, tensor<2x2xf32>) -> tensor<2x2xf32>} : () -> ()
}

// ----------------------

module {
  "polar_graph.node"() ( {
  ^bb0(%arg0: tensor<2x2xf32>):  // no predecessors
    %0 = "polar_graph.create_tensor"() {virtual_address = 33 : index} : () -> tensor<2x2xf32>
    "polar_graph.alloc"(%0) : (tensor<2x2xf32>) -> ()
    "polar_graph.lock"(%0) {lock_type = "read_write"} : (tensor<2x2xf32>) -> ()
    "polar_graph.lock"(%arg0) {lock_type = "read"} : (tensor<2x2xf32>) -> ()
    "polar_graph.copy"(%arg0, %0) : (tensor<2x2xf32>, tensor<2x2xf32>) -> ()
    "polar_graph.release"(%arg0) : (tensor<2x2xf32>) -> ()
    "polar_graph.release"(%0) : (tensor<2x2xf32>) -> ()
    polar_graph.return %0 : tensor<2x2xf32>
  }) {cluster_id = 1 : index, node_id = 9 : index, sym_name = "testCopy", type = (tensor<2x2xf32>) -> tensor<2x2xf32>} : () -> ()
}

// ----------------------

module {
  "polar_graph.node"() ( {
  ^bb0(%arg0: tensor<2x2xf32>, %arg1: tensor<2x2xf32>):  // no predecessors
    %0 = "polar_graph.create_tensor"() {virtual_address = 33 : index} : () -> tensor<2x2xf32>
    "polar_graph.alloc"(%0) : (tensor<2x2xf32>) -> ()
    "polar_graph.lock"(%0) {lock_type = "read_write"} : (tensor<2x2xf32>) -> ()
    "polar_graph.lock"(%arg1) {lock_type = "read"} : (tensor<2x2xf32>) -> ()
    "polar_graph.lock"(%arg0) {lock_type = "read"} : (tensor<2x2xf32>) -> ()
    "polar_graph.divide"(%arg0, %arg1, %0) : (tensor<2x2xf32>, tensor<2x2xf32>, tensor<2x2xf32>) -> ()
    "polar_graph.release"(%arg1) : (tensor<2x2xf32>) -> ()
    "polar_graph.release"(%arg0) : (tensor<2x2xf32>) -> ()
    "polar_graph.release"(%0) : (tensor<2x2xf32>) -> ()
    polar_graph.return %0 : tensor<2x2xf32>
  }) {cluster_id = 1 : index, node_id = 9 : index, sym_name = "testDivide", type = (tensor<2x2xf32>, tensor<2x2xf32>) -> tensor<2x2xf32>} : () -> ()
}

// ----------------------

module {
  "polar_graph.node"() ( {
  ^bb0(%arg0: tensor<2x2xf32>, %arg1: tensor<2x2xf32>):  // no predecessors
    %0 = "polar_graph.create_tensor"() {virtual_address = 33 : index} : () -> tensor<2x2xf32>
    "polar_graph.alloc"(%0) : (tensor<2x2xf32>) -> ()
    "polar_graph.lock"(%0) {lock_type = "read_write"} : (tensor<2x2xf32>) -> ()
    "polar_graph.lock"(%arg1) {lock_type = "read"} : (tensor<2x2xf32>) -> ()
    "polar_graph.lock"(%arg0) {lock_type = "read"} : (tensor<2x2xf32>) -> ()
    "polar_graph.logloss"(%arg0, %arg1, %0) : (tensor<2x2xf32>, tensor<2x2xf32>, tensor<2x2xf32>) -> ()
    "polar_graph.release"(%arg1) : (tensor<2x2xf32>) -> ()
    "polar_graph.release"(%arg0) : (tensor<2x2xf32>) -> ()
    "polar_graph.release"(%0) : (tensor<2x2xf32>) -> ()
    polar_graph.return %0 : tensor<2x2xf32>
  }) {cluster_id = 1 : index, node_id = 9 : index, sym_name = "testDivide", type = (tensor<2x2xf32>, tensor<2x2xf32>) -> tensor<2x2xf32>} : () -> ()
}

// ----------------------

module {
  "polar_graph.node"() ( {
  ^bb0(%arg0: tensor<2x2xf32>, %arg1: tensor<2x2xf32>):  // no predecessors
    %0 = "polar_graph.create_tensor"() {virtual_address = 33 : index} : () -> tensor<2x2xf32>
    "polar_graph.alloc"(%0) : (tensor<2x2xf32>) -> ()
    "polar_graph.lock"(%0) {lock_type = "read_write"} : (tensor<2x2xf32>) -> ()
    "polar_graph.lock"(%arg1) {lock_type = "read"} : (tensor<2x2xf32>) -> ()
    "polar_graph.lock"(%arg0) {lock_type = "read"} : (tensor<2x2xf32>) -> ()
    "polar_graph.matmul"(%arg0, %arg1, %0) {transpose_left = false, transpose_right = true} : (tensor<2x2xf32>, tensor<2x2xf32>, tensor<2x2xf32>) -> ()
    "polar_graph.release"(%arg1) : (tensor<2x2xf32>) -> ()
    "polar_graph.release"(%arg0) : (tensor<2x2xf32>) -> ()
    "polar_graph.release"(%0) : (tensor<2x2xf32>) -> ()
    polar_graph.return %0 : tensor<2x2xf32>
  }) {cluster_id = 1 : index, node_id = 9 : index, sym_name = "testMatmul", type = (tensor<2x2xf32>, tensor<2x2xf32>) -> tensor<2x2xf32>} : () -> ()
}

// ----------------------

module {
  "polar_graph.node"() ( {
  ^bb0(%arg0: tensor<2x2xf32>, %arg1: tensor<2x2xf32>):  // no predecessors
    %0 = "polar_graph.create_tensor"() {virtual_address = 33 : index} : () -> tensor<2x2xf32>
    "polar_graph.alloc"(%0) : (tensor<2x2xf32>) -> ()
    "polar_graph.lock"(%0) {lock_type = "read_write"} : (tensor<2x2xf32>) -> ()
    "polar_graph.lock"(%arg1) {lock_type = "read"} : (tensor<2x2xf32>) -> ()
    "polar_graph.lock"(%arg0) {lock_type = "read"} : (tensor<2x2xf32>) -> ()
    "polar_graph.mul"(%arg0, %arg1, %0) : (tensor<2x2xf32>, tensor<2x2xf32>, tensor<2x2xf32>) -> ()
    "polar_graph.release"(%arg1) : (tensor<2x2xf32>) -> ()
    "polar_graph.release"(%arg0) : (tensor<2x2xf32>) -> ()
    "polar_graph.release"(%0) : (tensor<2x2xf32>) -> ()
    polar_graph.return %0 : tensor<2x2xf32>
  }) {cluster_id = 1 : index, node_id = 9 : index, sym_name = "testMul", type = (tensor<2x2xf32>, tensor<2x2xf32>) -> tensor<2x2xf32>} : () -> ()
}

// ----------------------

module {
  "polar_graph.node"() ( {
  ^bb0(%arg0: tensor<2x2xf32>, %arg1: tensor<2x2xf32>):  // no predecessors
    %0 = "polar_graph.create_tensor"() {virtual_address = 33 : index} : () -> tensor<2x2xf32>
    "polar_graph.alloc"(%0) : (tensor<2x2xf32>) -> ()
    "polar_graph.lock"(%0) {lock_type = "read_write"} : (tensor<2x2xf32>) -> ()
    "polar_graph.lock"(%arg1) {lock_type = "read"} : (tensor<2x2xf32>) -> ()
    "polar_graph.lock"(%arg0) {lock_type = "read"} : (tensor<2x2xf32>) -> ()
    "polar_graph.mulconcat"(%arg0, %arg1, %0) : (tensor<2x2xf32>, tensor<2x2xf32>, tensor<2x2xf32>) -> ()
    "polar_graph.release"(%arg1) : (tensor<2x2xf32>) -> ()
    "polar_graph.release"(%arg0) : (tensor<2x2xf32>) -> ()
    "polar_graph.release"(%0) : (tensor<2x2xf32>) -> ()
    polar_graph.return %0 : tensor<2x2xf32>
  }) {cluster_id = 1 : index, node_id = 9 : index, sym_name = "testMulconcat", type = (tensor<2x2xf32>, tensor<2x2xf32>) -> tensor<2x2xf32>} : () -> ()
}

// ----------------------

module {
  "polar_graph.node"() ( {
  ^bb0(%arg0: tensor<2x2xf32>):  // no predecessors
    %0 = "polar_graph.create_tensor"() {virtual_address = 33 : index} : () -> tensor<2x2xf32>
    "polar_graph.alloc"(%0) : (tensor<2x2xf32>) -> ()
    "polar_graph.lock"(%0) {lock_type = "read_write"} : (tensor<2x2xf32>) -> ()
    "polar_graph.lock"(%arg0) {lock_type = "read"} : (tensor<2x2xf32>) -> ()
    "polar_graph.sigmoid"(%arg0, %0) : (tensor<2x2xf32>, tensor<2x2xf32>) -> ()
    "polar_graph.release"(%arg0) : (tensor<2x2xf32>) -> ()
    "polar_graph.release"(%0) : (tensor<2x2xf32>) -> ()
    polar_graph.return %0 : tensor<2x2xf32>
  }) {cluster_id = 1 : index, node_id = 9 : index, sym_name = "testSigmoid", type = (tensor<2x2xf32>) -> tensor<2x2xf32>} : () -> ()
}

// ----------------------

module {
  "polar_graph.node"() ( {
  ^bb0(%arg0: tensor<2x2xf32>):  // no predecessors
    %0 = "polar_graph.create_tensor"() {virtual_address = 33 : index} : () -> tensor<2x2xf32>
    "polar_graph.alloc"(%0) : (tensor<2x2xf32>) -> ()
    "polar_graph.lock"(%0) {lock_type = "read_write"} : (tensor<2x2xf32>) -> ()
    "polar_graph.lock"(%arg0) {lock_type = "read"} : (tensor<2x2xf32>) -> ()
    "polar_graph.transpose"(%arg0, %0) : (tensor<2x2xf32>, tensor<2x2xf32>) -> ()
    "polar_graph.release"(%arg0) : (tensor<2x2xf32>) -> ()
    "polar_graph.release"(%0) : (tensor<2x2xf32>) -> ()
    polar_graph.return %0 : tensor<2x2xf32>
  }) {cluster_id = 1 : index, node_id = 9 : index, sym_name = "testTranspose", type = (tensor<2x2xf32>) -> tensor<2x2xf32>} : () -> ()
}

// ----------------------

module {
  "polar_graph.node"() ( {
  ^bb0:  // no predecessors
    %0 = "polar_graph.create_tensor"() {virtual_address = 33 : index} : () -> tensor<2x2xf32>
    %cst = constant 1.000000e+0 : f32
    "polar_graph.alloc"(%0) : (tensor<2x2xf32>) -> ()
    "polar_graph.lock"(%0) {lock_type = "read_write"} : (tensor<2x2xf32>) -> ()
    "polar_graph.fill"(%cst, %0) : (f32, tensor<2x2xf32>) -> ()
    "polar_graph.release"(%0) : (tensor<2x2xf32>) -> ()
    polar_graph.return %0 : tensor<2x2xf32>
  }) {cluster_id = 1 : index, node_id = 9 : index, sym_name = "testFill", type = () -> tensor<2x2xf32>} : () -> ()
}
