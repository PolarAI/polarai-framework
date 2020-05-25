file(GLOB_RECURSE CORE_TEST_SRC "*.cpp")

add_athena_executable(TestUnitCoreRunnable ${CORE_TEST_SRC})
target_include_directories(TestUnitCoreRunnable PUBLIC $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/checkers/include>
        $<INSTALL_INTERFACE:checkers/include)
target_link_libraries(TestUnitCoreRunnable PUBLIC
        AthenaCore
        AthenaOperation
        AthenaLoaders
        AthenaUtils
        Threads::Threads
        gtest gtest_main
        AthenaDep::googletest)

add_test(NAME CoreUnitTest COMMAND TestUnitCoreRunnable)
