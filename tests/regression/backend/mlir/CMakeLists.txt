file(GLOB_RECURSE REG_TEST_SRC "*.cpp")

include(AthenaIntegrationTest)

find_package(MLIR COMPONENTS IR Support Parser Analysis)
find_package(LLVM CONFIG)

llvm_map_components_to_libnames(llvm_libs
        core
        irreader
        support
        transformutils
        passes
        codegen
        orcjit
        AllTargetsCodeGens AllTargetsAsmParsers AllTargetsDescs AllTargetsInfos
        lto)

add_athena_integration_test(
        TARGET_NAME MLIRRegression
        SRCS ${REG_TEST_SRC}
        LIBS athena AthenaDep::effcee ${MLIR_LIBS} ${ATH_BACKEND_LLVM_MLIR} ${llvm_libs})

target_include_directories(TestIntegrationMLIRRegressionRunnable PRIVATE ${MLIR_INCLUDE_DIRS})
