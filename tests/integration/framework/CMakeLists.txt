add_library(IntegrationTestFramework OBJECT framework.cpp)

configure_file(options.h.in options.h)

target_include_directories(IntegrationTestFramework PUBLIC ${CMAKE_CURRENT_BINARY_DIR})
target_link_libraries(IntegrationTestFramework PUBLIC
        AthenaDep::yaml-cpp
        AthenaDep::googletest-gtest)
if (UNIX AND NOT APPLE)
    if ("${CMAKE_CXX_COMPILER_ID}" MATCHES "GNU")
        target_link_libraries(IntegrationTestFramework PUBLIC
            $<$<AND:$<CXX_COMPILER_ID:GNU>,$<VERSION_LESS:$<CXX_COMPILER_VERSION>,9.0>>:stdc++fs>)
    elseif("${CMAKE_CXX_COMPILER_ID}" MATCHES "GNU")
        target_link_libraries(IntegrationTestFramework PUBLIC stdc++fs)
    endif()
endif()
