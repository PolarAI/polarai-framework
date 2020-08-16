file(GLOB PATTERNS_SRC "Patterns/*.cpp")

add_library(PolarScript SHARED
  CompilerFrontend.cpp
  KernelAttrInfo.cpp
  MlirGen.cpp
  CodeGenAction.cpp
  Pattern.cpp
  ${PATTERNS_SRC}
)

target_link_directories(PolarScript PRIVATE ${LLVM_BUILD_LIBRARY_DIR})
llvm_map_components_to_libnames(llvm_libs
        core
        support
        ExecutionEngine
        MC
        MCJIT
        Object
        OrcJit
        Option
        RuntimeDyld
        native
)

target_link_libraries(PolarScript PUBLIC
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
    ${llvm_libs}
    MLIRGPU
    MLIRIR
)

if (UNIX)
  target_compile_options(PolarScript PRIVATE -fno-rtti -fno-exceptions)
endif()
