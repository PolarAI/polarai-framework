file(GLOB_RECURSE REG_TEST_SRC "*.cpp")

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

add_athena_executable(
        TestRegressionMLIRRunnable
        ${REG_TEST_SRC})

target_include_directories(TestRegressionMLIRRunnable PRIVATE ${LLVM_INCLUDE_DIRS})
get_property(dialect_libs GLOBAL PROPERTY MLIR_DIALECT_LIBS)
get_property(conversion_libs GLOBAL PROPERTY MLIR_CONVERSION_LIBS)
target_link_libraries(TestRegressionMLIRRunnable PRIVATE
        athena
        ${dialect_libs}
        ${conversion_libs}
        AthenaDep::googletest
        AthenaDep::effcee)

add_test(NAME MLIRRegressionTest COMMAND TestRegressionMLIRRunnable)
