add_athena_library(athena-model STATIC NativeModel.cpp)

include(AthenaCompileProtobuf)

athena_add_protobuf_module(LIB_NAME athena-graph-proto
        PROTO_DIR ${PROJECT_SOURCE_DIR}/utils/graph_proto)
target_link_libraries(athena-model PUBLIC athena-graph-proto)