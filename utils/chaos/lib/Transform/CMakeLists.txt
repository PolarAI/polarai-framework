find_package(MLIR COMPONENTS
        IR TargetLLVMIR Dialect LLVMIR Support Translation Pass Transforms Analysis AffineOps StandardOps
        TransformUtils LoopOps TargetLLVMIRModuleTranslation AffineToStandard VectorOps VectorToLLVM
        StandardToLLVM LoopToStandard)

add_library(
        chaosTransform
        STATIC
        IRTransformer.cpp
        LoopAffinitizer.cpp
        LLVMToStandardTypeConverter.cpp
        LLVMToStandardConversionPattern.cpp
        LLVMToStdPass.cpp
        ChaosDialect.cpp
        ChaosLoweringPass.cpp
        IRCleanUpPass.cpp
)

file(MAKE_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}/tablegen)

add_custom_command(OUTPUT ${CMAKE_CURRENT_BINARY_DIR}/tablegen/ChaosDialect.h.inc
        ${CMAKE_CURRENT_BINARY_DIR}/tablegen/ChaosDialect.h.fake
        COMMAND ${MLIR_TBLGEN} --gen-op-decls -o=${CMAKE_CURRENT_BINARY_DIR}/tablegen/ChaosDialect.h.inc
        -I=${MLIR_INCLUDE_DIRS} ${CMAKE_CURRENT_SOURCE_DIR}/ChaosDialect.td
        )

add_custom_command(OUTPUT ${CMAKE_CURRENT_BINARY_DIR}/tablegen/ChaosDialect.cpp.inc
        ${CMAKE_CURRENT_BINARY_DIR}/tablegen/ChaosDialect.cpp.fake
        COMMAND ${MLIR_TBLGEN} --gen-op-defs -o=${CMAKE_CURRENT_BINARY_DIR}/tablegen/ChaosDialect.cpp.inc
        -I=${MLIR_INCLUDE_DIRS} ${CMAKE_CURRENT_SOURCE_DIR}/ChaosDialect.td
        )

add_custom_target(chaos_dialect_tblgen
        DEPENDS ${CMAKE_CURRENT_BINARY_DIR}/tablegen/ChaosDialect.h.inc
        ${CMAKE_CURRENT_BINARY_DIR}/tablegen/ChaosDialect.h.fake
        ${CMAKE_CURRENT_BINARY_DIR}/tablegen/ChaosDialect.cpp.inc
        ${CMAKE_CURRENT_BINARY_DIR}/tablegen/ChaosDialect.cpp.fake
        SOURCES ${CMAKE_CURRENT_SOURCE_DIR}/ChaosDialect.td)

add_dependencies(chaosTransform chaos_dialect_tblgen)
target_include_directories(chaosTransform PRIVATE ${CMAKE_CURRENT_BINARY_DIR}/tablegen/)


athena_disable_exceptions(chaosTransform)
athena_disable_rtti(chaosTransform)
if (UNIX)
    target_compile_options(chaosTransform PRIVATE -fPIC)
endif ()

generate_export_header(
        chaosTransform
        EXPORT_MACRO_NAME CHAOS_TRANSFORM_EXPORT
        EXPORT_FILE_NAME ${CMAKE_BINARY_DIR}/include/chaos/Transform/export.h
)

llvm_map_components_to_libnames(llvm_libs
        core
        irreader
        support
        TransformUtils
        bitwriter
        passes)

# MLIR is full of weak symbols. Ubuntu is not doing well with them, so
# this is a hack to overcome the issue. Apparently, it produces lots of
# warning, but at least makes possible further development.
if (UNIX AND NOT APPLE)
    set(WHOLE_ARCHIVE -Wl,--exclude-libs,ALL -Wl,--whole-archive)
    set(NOWHOLE_ARCHIVE -Wl,--no-whole-archive)
endif ()

target_link_libraries(chaosTransform PRIVATE ${WHOLE_ARCHIVE} ${MLIR_LIBS} ${NOWHOLE_ARCHIVE} ${llvm_libs})
target_include_directories(chaosTransform PUBLIC ${LLVM_INCLUDE_DIRS} PUBLIC ${MLIR_INCLUDE_DIRS})
target_compile_definitions(chaosTransform PUBLIC ${LLVM_DEFINITIONS})
target_include_directories(chaosTransform PUBLIC ${CMAKE_SOURCE_DIR}/utils/chaos/include)
target_include_directories(chaosTransform PUBLIC ${CMAKE_BINARY_DIR}/include/chaos)