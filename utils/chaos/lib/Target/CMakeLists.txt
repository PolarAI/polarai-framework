find_package(LLVM CONFIG)

add_athena_library(
        chaosTarget
        SHARED
        CHAOS_TARGET
        chaos/target.h
        ObjectEmitter.cpp
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

target_include_directories(chaosTarget PRIVATE ${LLVM_INCLUDE_DIRS})
target_compile_definitions(chaosTarget PRIVATE ${LLVM_DEFINITIONS})
target_link_libraries(chaosTarget PRIVATE ${llvm_libs})
target_include_directories(chaosTarget PRIVATE ${CMAKE_SOURCE_DIR}/utils/chaos/include)