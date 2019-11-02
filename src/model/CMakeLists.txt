set(ATH_MODEL "AthenaModel" CACHE STRING "" FORCE)

add_athena_library(${ATH_MODEL} STATIC NativeModel.cpp)

include(AthenaCompileProtobuf)

athena_add_protobuf_module(LIB_NAME athena-graph-proto
        PROTO_DIR ${PROJECT_SOURCE_DIR}/utils/graph_proto)
target_link_libraries(${ATH_MODEL} PUBLIC athena-graph-proto)