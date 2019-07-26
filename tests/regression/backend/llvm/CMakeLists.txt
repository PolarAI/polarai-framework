file(GLOB_RECURSE REG_TEST_SRC "*.cpp")

add_executable(TestRegressionBackendLlvmRunnable ${REG_TEST_SRC})

list(REMOVE_ITEM REG_TEST_SRC "${CMAKE_CURRENT_SOURCE_DIR}/main.cpp")

set(TESTS_LIST "")
foreach (TEST ${REG_TEST_SRC})
    set(TESTS_LIST "${TESTS_LIST}${TEST}\n")
endforeach ()

file(WRITE ${CMAKE_CURRENT_BINARY_DIR}/tests.txt ${TESTS_LIST})

find_program(PYTHON python python3)

add_test(NAME TestRegressionBackendLlvm
        COMMAND ${PYTHON} ${CMAKE_CURRENT_SOURCE_DIR}/runner.py ${CMAKE_CURRENT_BINARY_DIR}/tests.txt ${FILE}
        )
