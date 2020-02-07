find_package(Python COMPONENTS Interpreter)

# for easy IDE integration only
add_custom_target(RunChaosRegression
        COMMAND ${Python_EXECUTABLE} ${CMAKE_SOURCE_DIR}/utils/test_runner/runner.py
        --bin-path="${CMAKE_BINARY_DIR}" --llvm-path="$ENV{LLVM_DIR}"
        ${CMAKE_CURRENT_SOURCE_DIR})
add_test(
        NAME ChaosRegressionTests
        COMMAND ${Python_EXECUTABLE} ${CMAKE_SOURCE_DIR}/utils/test_runner/runner.py
        --bin-path="${CMAKE_BINARY_DIR}" --llvm-path="$ENV{LLVM_DIR}"
        ${CMAKE_CURRENT_SOURCE_DIR}
)