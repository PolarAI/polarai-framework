set(ATH_MODEL "AthenaModel" CACHE STRING "" FORCE)

add_athena_library(${ATH_MODEL} STATIC
        ATH_MODEL # export name
        model/model_export.h # export file name
        NativeModel.cpp
        DotModel.cpp)

include(AthenaCompileProtobuf)

athena_add_protobuf_module(LIB_NAME athena_graph_proto
        PROTO_DIR ${PROJECT_SOURCE_DIR}/utils/graph_proto)
target_link_libraries(${ATH_MODEL} PUBLIC athena_graph_proto)