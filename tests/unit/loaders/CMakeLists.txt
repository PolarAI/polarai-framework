add_athena_executable(runUnitLoaders MemoryLoader.cpp)
target_link_libraries(runUnitLoaders athena-core)
target_link_libraries(runUnitLoaders backend-llvm)
target_link_libraries(runUnitLoaders MemoryLoader)
target_link_libraries(runUnitLoaders ${GTEST_BOTH_LIBRARIES} Threads::Threads)

add_test(NAME UnitLoaders COMMAND runUnitLoaders)

install(TARGETS runUnitLoaders
        RUNTIME DESTINATION test)
