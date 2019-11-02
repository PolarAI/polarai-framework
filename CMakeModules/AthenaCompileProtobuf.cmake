function(athena_add_protobuf_module)
    cmake_parse_arguments(PARSED "" "LIB_NAME;PROTO_DIR" "" ${ARGN})

    find_package(Protobuf REQUIRED)

    file(GLOB SRC ${PARSED_PROTO_DIR}/*.proto)

    PROTOBUF_GENERATE_CPP(PROTO_SRCS PROTO_HDRS ${SRC})

    add_athena_library(${PARSED_LIB_NAME} OBJECT
            ${PARSED_LIB_NAME}
            ../${PARSED_LIB_NAME}_export.h
            ${PROTO_SRCS})
    target_compile_definitions(${PARSED_LIB_NAME} PUBLIC GOOGLE_PROTOBUF_NO_RTTI)
    target_link_libraries(${PARSED_LIB_NAME} PRIVATE ${PROTOBUF_LIBRARIES})
    target_include_directories(${PARSED_LIB_NAME}
            PUBLIC $<BUILD_INTERFACE:${CMAKE_CURRENT_BINARY_DIR}>
            ${PROTOBUF_INCLUDE_DIR})

    export(TARGETS ${PARSED_LIB_NAME} FILE AthenaTarget.cmake)
endfunction(athena_add_protobuf_module)