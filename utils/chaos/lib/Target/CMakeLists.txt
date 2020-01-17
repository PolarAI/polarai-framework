find_package(LLVM CONFIG)

add_library(
        chaosTarget
        STATIC
        ObjectEmitter.cpp
)

athena_disable_exceptions(chaosTarget)
athena_disable_rtti(chaosTarget)
if (UNIX)
    target_compile_options(chaosTarget PRIVATE -fPIC)
endif ()

generate_export_header(
        chaosTarget
        EXPORT_MACRO_NAME CHAOS_TARGET_EXPORT
        EXPORT_FILE_NAME ${CMAKE_BINARY_DIR}/include/chaos/Target/export.h
)

llvm_map_components_to_libnames(llvm_libs
        core
        irreader
        support
        TransformUtils
        bitwriter
        passes
        analysis
        codegen
        lto
        AllTargetsCodeGens AllTargetsAsmPrinters AllTargetsAsmParsers AllTargetsDescs AllTargetsInfos)

target_include_directories(chaosTarget PUBLIC ${LLVM_INCLUDE_DIRS})
target_compile_definitions(chaosTarget PUBLIC ${LLVM_DEFINITIONS})
target_link_libraries(chaosTarget PUBLIC ${llvm_libs})
target_include_directories(chaosTarget PUBLIC ${CMAKE_SOURCE_DIR}/utils/chaos/include)
target_include_directories(chaosTarget PUBLIC ${CMAKE_BINARY_DIR}/include/chaos)