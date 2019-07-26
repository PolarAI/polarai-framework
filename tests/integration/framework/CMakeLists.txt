add_library(IntegrationTestFramework OBJECT framework.cpp)

configure_file(options.h.in options.h)

target_include_directories(IntegrationTestFramework PUBLIC ${CMAKE_CURRENT_BINARY_DIR})
target_link_libraries(IntegrationTestFramework PUBLIC
        AthenaDep::yaml-cpp
        AthenaDep::Boost
        AthenaDep::googletest-gtest)
