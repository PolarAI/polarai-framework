add_athena_executable(chaoscc main.cpp)

target_link_libraries(chaoscc PRIVATE chaosDriver)
target_include_directories(chaoscc PRIVATE ${CMAKE_SOURCE_DIR}/utils/chaos/include)