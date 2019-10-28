function(add_rt_unit_test NAME RT)
    file(GLOB_RECURSE UNIT_TEST_SRC "*.cpp")

    add_athena_executable(TestUnit${NAME}Runnable ${UNIT_TEST_SRC})
    target_link_libraries(TestUnit${NAME}Runnable PRIVATE
            athena
            ${RT}
            Threads::Threads
            AthenaDep::googletest)

    add_test(NAME Core${NAME}Test COMMAND TestUnit${NAME}Runnable)
endfunction(add_rt_unit_test)

add_rt_unit_test(CPU runtime-llvm-cpu)