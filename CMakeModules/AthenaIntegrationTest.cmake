set(INTEG_RUN_SCRIPT_NAME "integration_test_run.py")

function(add_athena_integration_test)
    cmake_parse_arguments(PARSED_ARGS "" "TARGET_NAME" "SRCS;LIBS" ${ARGN})
    set(INTEG_CONFIG_INNER_NAME "config.yml")
    set(INTEG_CONFIG_OUTER_NAME "${PARSED_ARGS_TARGET_NAME}.yml")
    add_executable(${PARSED_ARGS_TARGET_NAME} ${modifier} ${PARSED_ARGS_SRCS} $<TARGET_OBJECTS:IntegrationTestFramework>)
    find_package(Boost 1.50.0 COMPONENTS filesystem REQUIRED QUIET)
    find_package(YAMLCPP)

    target_link_libraries(${PARSED_ARGS_TARGET_NAME} ${GTEST_LIBRARY} Threads::Threads athena ${YAMLCPP_LIBRARY})

    target_link_libraries(${PARSED_ARGS_TARGET_NAME} ${Boost_LIBRARIES})
    if (PARSED_ARGS_LIBS)
        target_link_libraries(${PARSED_ARGS_TARGET_NAME} ${PARSED_ARGS_LIBS})
    endif ()
    if (EXISTS ${CMAKE_CURRENT_SOURCE_DIR}/${INTEG_CONFIG_INNER_NAME})
        configure_file(${INTEG_CONFIG_INNER_NAME} ${INTEG_CONFIG_OUTER_NAME})
        install(FILES ${CMAKE_CURRENT_BINARY_DIR}/${INTEG_CONFIG_OUTER_NAME} DESTINATION test)
    endif ()
    add_test(NAME "${PARSED_ARGS_TARGET_NAME}IntegrationTest" COMMAND ${PARSED_ARGS_TARGET_NAME})

    install(TARGETS ${PARSED_ARGS_TARGET_NAME}
            RUNTIME DESTINATION test)
endfunction(add_athena_integration_test)
