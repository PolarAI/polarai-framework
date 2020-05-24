// RUN: athena-opt --legalize-barriers %s | FileCheck %s
module {
  func @inputA(%arg0: !ath_rt.graph_handle) -> !ath_rt.event attributes {cluster_id = 0 : index, node_id = 0 : index} {
    %1 = "ath_rt.null_event"() : () -> !ath_rt.event
    return %1 : !ath_rt.event
  }
  func @inputB(%arg0: !ath_rt.graph_handle) -> !ath_rt.event attributes {cluster_id = 0 : index, node_id = 1 : index} {
    %1 = "ath_rt.null_event"() : () -> !ath_rt.event
    return %1 : !ath_rt.event
  }
  func @sum(%arg0: !ath_rt.graph_handle) -> !ath_rt.event attributes {cluster_id = 1 : index, node_id = 2 : index} {
    %1 = "ath_rt.null_event"() : () -> !ath_rt.event
    return %1 : !ath_rt.event
  }
  func @mainGraph(%arg0: !ath_rt.graph_handle) {
    %0 = call @inputA(%arg0) : (!ath_rt.graph_handle) -> !ath_rt.event
    %1 = call @inputB(%arg0) : (!ath_rt.graph_handle) -> !ath_rt.event
    "ath_rt.barrier"() {cluster_id = 0 : index} : () -> ()
    %2 = call @sum(%arg0) : (!ath_rt.graph_handle) -> !ath_rt.event
    return
  }
}

// CHECK: module {
// CHECK-NEXT: func @inputA(%arg0: !ath_rt.graph_handle) -> !ath_rt.event attributes {cluster_id = 0 : index, node_id = 0 : index} {
// CHECK-NEXT: %0 = "ath_rt.null_event"() : () -> !ath_rt.event
// CHECK-NEXT: return %0 : !ath_rt.event
// CHECK-NEXT: }
// CHECK-NEXT: func @inputB(%arg0: !ath_rt.graph_handle) -> !ath_rt.event attributes {cluster_id = 0 : index, node_id = 1 : index} {
// CHECK-NEXT: %0 = "ath_rt.null_event"() : () -> !ath_rt.event
// CHECK-NEXT: return %0 : !ath_rt.event
// CHECK-NEXT: }
// CHECK-NEXT: func @sum(%arg0: !ath_rt.graph_handle) -> !ath_rt.event attributes {cluster_id = 1 : index, node_id = 2 : index} {
// CHECK-NEXT: %0 = "ath_rt.null_event"() : () -> !ath_rt.event
// CHECK-NEXT: return %0 : !ath_rt.event
// CHECK-NEXT: }
// CHECK-NEXT: func @mainGraph(%arg0: !ath_rt.graph_handle) {
// CHECK-NEXT: %0 = call @inputA(%arg0) : (!ath_rt.graph_handle) -> !ath_rt.event
// CHECK-NEXT: %1 = call @inputB(%arg0) : (!ath_rt.graph_handle) -> !ath_rt.event
// CHECK-NEXT: "ath_rt.barrier"(%0, %1) : (!ath_rt.event, !ath_rt.event) -> ()
// CHECK-NEXT: %2 = call @sum(%arg0) : (!ath_rt.graph_handle) -> !ath_rt.event
// CHECK-NEXT: return
// CHECK-NEXT: }
// CHECK-NEXT: }
