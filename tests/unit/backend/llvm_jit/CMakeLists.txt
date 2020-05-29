if(POLICY CMP0068)
    cmake_policy(SET CMP0068 NEW)
    set(CMAKE_BUILD_WITH_INSTALL_NAME_DIR ON)
endif()

if(POLICY CMP0075)
    cmake_policy(SET CMP0075 NEW)
endif()

if(POLICY CMP0077)
    cmake_policy(SET CMP0077 NEW)
endif()

find_package(LLVM REQUIRED CONFIG)
find_package(MLIR REQUIRED CONFIG)

get_property(dialect_libs GLOBAL PROPERTY MLIR_DIALECT_LIBS)
get_property(conversion_libs GLOBAL PROPERTY MLIR_CONVERSION_LIBS)

add_athena_executable(TestUnitJITRunnable 
                      AthenaJIT.cpp
                      $<TARGET_OBJECTS:${AthenaJitTarget}>)

llvm_map_components_to_libnames(llvm_libs
        core
        irreader
        support
        transformutils
        passes
        codegen
        orcjit
        AllTargetsCodeGens AllTargetsAsmParsers AllTargetsDescs AllTargetsInfos)
target_link_libraries(TestUnitJITRunnable PRIVATE
                      ${llvm_libs}
                      ${dialect_libs}
                      ${conversion_libs}
                      MLIRIR
                      MLIRPass
                      MLIRAthenaGraph
                      MLIRAthenaRuntime
                      MLIRAthenaPasses
                      MLIRAthenaConversion
                      AthenaDep::googletest)
target_include_directories(TestUnitJITRunnable PRIVATE
                           ${LLVM_INCLUDE_DIRS}
                           ${PROJECT_BINARY_DIR}/utils/dialects/include
                           ${PROJECT_SOURCE_DIR}/utils/dialects/include)
