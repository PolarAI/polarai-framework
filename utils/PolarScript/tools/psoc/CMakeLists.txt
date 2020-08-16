get_property(dialect_libs GLOBAL PROPERTY MLIR_DIALECT_LIBS)
get_property(conversion_libs GLOBAL PROPERTY MLIR_CONVERSION_LIBS)

add_llvm_executable(psoc main.cpp)

llvm_update_compile_flags(psoc)

llvm_map_components_to_libnames(llvm_libs
        core
        support
        Option
)
target_link_libraries(psoc PUBLIC 
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
    MLIRGPU
    MLIRIR
    PolarScript
    clangFrontend
        ${dialect_libs}
        ${conversion_libs}
)

if (UNIX)
  target_compile_options(psoc PRIVATE -fno-rtti -fno-exceptions)
endif()
