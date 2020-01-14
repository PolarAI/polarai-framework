find_package(LLVM CONFIG)
find_package(MLIR COMPONENTS
        IR TargetLLVMIR Dialect LLVMIR Support Translation Pass Transforms Analysis AffineOps StandardOps
        TransformUtils LoopOps TargetLLVMIRModuleTranslation AffineToStandard VectorOps VectorToLLVM)

add_athena_library(
        chaosTransform
        SHARED
        CHAOS_TRANSFORM
        chaos/transform.h
        IRTransformer.cpp
)

llvm_map_components_to_libnames(llvm_libs
        core
        irreader
        support
        TransformUtils
        bitwriter
        passes)

message(STATUS "${llvm_libs}")

target_link_libraries(chaosTransform PUBLIC ${llvm_libs} Athena::MLIR)
target_include_directories(chaosTransform PRIVATE ${LLVM_INCLUDE_DIRS} PUBLIC ${MLIR_INCLUDE_DIRS})
target_compile_definitions(chaosTransform PRIVATE ${LLVM_DEFINITIONS})
target_include_directories(chaosTransform PRIVATE ${CMAKE_SOURCE_DIR}/utils/chaos/include)