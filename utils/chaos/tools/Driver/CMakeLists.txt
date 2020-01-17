add_executable(chaoscc main.cpp)

if (UNIX AND NOT APPLE)
    athena_add_linker_options(chaoscc -Wl,--warn-unresolved-symbols)
endif ()
target_link_libraries(chaoscc PRIVATE chaosDriver)
target_include_directories(chaoscc PRIVATE ${CMAKE_SOURCE_DIR}/utils/chaos/include)