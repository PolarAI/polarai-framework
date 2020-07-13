include_directories(${PROJECT_BINARY_DIR}/src/core)
include_directories(${PROJECT_BINARY_DIR}/src/io)
include_directories(${PROJECT_BINARY_DIR}/src/loaders)
include_directories(${PROJECT_BINARY_DIR}/src/operation)
include_directories(${PROJECT_BINARY_DIR}/src/utils)
include_directories(${PROJECT_BINARY_DIR}/src/backend/generic/lib)

find_package(Threads)

add_subdirectory(unit)
add_subdirectory(integration)
