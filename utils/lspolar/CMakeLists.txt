include_directories(${PROJECT_BINARY_DIR}/src/utils)
add_polar_executable(lspolar main.cpp)

target_include_directories(lspolar PRIVATE 
                          ${PROJECT_SOURCE_DIR}/src/backend/generic/runtime/driver)

target_link_libraries(lspolar PRIVATE PolarAI::generic_backend_driver)
get_property(AVAILABLE_RT GLOBAL PROPERTY POLAR_AVAILABLE_RUNTIMES)
if (AVAILABLE_RT)
  add_dependencies(lspolar ${AVAILABLE_RT})
endif()
