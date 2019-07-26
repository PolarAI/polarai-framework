file(GLOB_RECURSE CORE_TEST_SRC "*.cpp")

add_athena_executable(runUnitCore ${CORE_TEST_SRC})
target_link_libraries(runUnitCore PRIVATE
        athena-core
        Threads::Threads
        AthenaDep::googletest)

add_test(NAME UnitCoreTests COMMAND runUnitCore)

install(TARGETS runUnitCore
        RUNTIME DESTINATION test)