find_package(MLIR COMPONENTS
        IR TargetLLVMIR Dialect LLVMIR Support Translation Pass Transforms Analysis AffineOps StandardOps
        TransformUtils LoopOps TargetLLVMIRModuleTranslation AffineToStandard VectorOps VectorToLLVM
        StandardToLLVM LoopToStandard)

add_library(
        chaosFrontend
        STATIC
        Frontend.cpp
        CXX/CXXFrontend.cpp
        CXX/mlir/MLIRGen.cpp
        CXX/mlir/MLIRASTConsumer.cpp
        CXX/mlir/TypeConverter.cpp
)

athena_disable_exceptions(chaosFrontend)
athena_disable_rtti(chaosFrontend)
if (UNIX)
    target_compile_options(chaosFrontend PRIVATE -fPIC)
endif ()

generate_export_header(
        chaosFrontend
        EXPORT_MACRO_NAME CHAOS_FRONTEND_EXPORT
        EXPORT_FILE_NAME ${CMAKE_BINARY_DIR}/include/chaos/Frontend/export.h
)

llvm_map_components_to_libnames(llvm_libs
        core
        irreader
        support
        TransformUtils
        bitwriter
        MC MCA MCParser
        ProfileData
        FrontendOpenMP
        Option
        passes
        ObjCARCOpts
        Coroutines
        LTO
        Coverage
        AllTargetsCodeGens AllTargetsAsmPrinters AllTargetsAsmParsers AllTargetsDescs AllTargetsInfos)

# MLIR is full of weak symbols. Ubuntu is not doing well with them, so
# this is a hack to overcome the issue. Apparently, it produces lots of
# warning, but at least makes possible further development.
if (UNIX AND NOT APPLE)
    set(WHOLE_ARCHIVE -Wl,--exclude-libs,ALL -Wl,--whole-archive)
    set(NOWHOLE_ARCHIVE -Wl,--no-whole-archive)
endif ()

target_include_directories(chaosFrontend PUBLIC ${LLVM_INCLUDE_DIRS})
target_compile_definitions(chaosFrontend PUBLIC ${LLVM_DEFINITIONS})
target_link_libraries(chaosFrontend PUBLIC
        chaosDialects
        ${WHOLE_ARCHIVE} ${MLIR_LIBS} ${NOWHOLE_ARCHIVE}
        ${llvm_libs})
target_include_directories(chaosFrontend PUBLIC ${CMAKE_SOURCE_DIR}/utils/chaos/include)
target_include_directories(chaosFrontend PUBLIC ${CMAKE_BINARY_DIR}/include/chaos)

# TODO better clang linking
target_link_directories(chaosFrontend PUBLIC $ENV{LLVM_DIR}/lib)
target_link_libraries(chaosFrontend PRIVATE
        clangBasic
        clangCodeGen
        clangDriver
        clangFrontend
        clangFrontendTool
        clangCodeGen
        clangRewriteFrontend
        clangARCMigrate
        clangStaticAnalyzerFrontend
        clangStaticAnalyzerCheckers
        clangStaticAnalyzerCore
        clangCrossTU
        clangIndex
        clangFrontend
        clangDriver
        clangParse
        clangSerialization
        clangSema
        clangAnalysis
        clangEdit
        clangFormat
        clangToolingInclusions
        clangToolingCore
        clangRewrite
        clangASTMatchers
        clangAST
        clangLex
        clangBasic
        clangAST)