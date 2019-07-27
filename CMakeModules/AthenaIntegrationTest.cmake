set(INTEG_RUN_SCRIPT_NAME "integration_test_run.py")

function(add_athena_integration_test)
    cmake_parse_arguments(PARSED_ARGS "" "TARGET_NAME" "SRCS;LIBS" ${ARGN})
    set(INTEG_CONFIG_INNER_NAME "config.yml")
    set(TARGET_NAME TestIntegration${PARSED_ARGS_TARGET_NAME}Runnable)
    set(INTEG_CONFIG_OUTER_NAME "${TARGET_NAME}.yml")
    add_executable(${TARGET_NAME} ${modifier} ${PARSED_ARGS_SRCS} $<TARGET_OBJECTS:IntegrationTestFramework>)

    target_link_libraries(${TARGET_NAME} PRIVATE
            Threads::Threads
            athena
            AthenaDep::googletest-gtest
            AthenaDep::Boost
            AthenaDep::yaml-cpp
            )

    if (PARSED_ARGS_LIBS)
        target_link_libraries(${TARGET_NAME} PRIVATE ${PARSED_ARGS_LIBS})
    endif ()
    if (EXISTS ${CMAKE_CURRENT_SOURCE_DIR}/${INTEG_CONFIG_INNER_NAME})
        configure_file(${INTEG_CONFIG_INNER_NAME} ${INTEG_CONFIG_OUTER_NAME})
    endif ()
    add_test(NAME "${TARGET_NAME}IntegrationTest" COMMAND ${TARGET_NAME})
endfunction(add_athena_integration_test)
