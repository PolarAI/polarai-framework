add_athena_executable(runUnitLoaders MemoryLoader.cpp)
target_link_libraries(runUnitLoaders PRIVATE
        athena-core
        backend-llvm
        MemoryLoader
        Threads::Threads
        AthenaDep::googletest)

add_test(NAME UnitLoaders COMMAND runUnitLoaders)

install(TARGETS runUnitLoaders
        RUNTIME DESTINATION test)
