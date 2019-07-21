file(GLOB_RECURSE CORE_TEST_SRC "*.cpp")

add_athena_executable(runUnitCore ${CORE_TEST_SRC})
target_link_libraries(runUnitCore athena-core)
target_link_libraries(runUnitCore ${GTEST_BOTH_LIBRARIES} Threads::Threads)
target_include_directories(runUnitCore PUBLIC ${GTEST_INCLUDE_DIRS})

add_test(NAME UnitCoreTests COMMAND runUnitCore)

install(TARGETS runUnitCore
        RUNTIME DESTINATION test)