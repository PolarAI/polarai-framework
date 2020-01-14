find_package(LLVM CONFIG)

add_athena_library(
        chaosDriver
        SHARED
        CHAOS_DRIVER
        chaos/driver.h
        Driver.cpp
)

llvm_map_components_to_libnames(llvm_libs
        core
        irreader
        support)

target_include_directories(chaosDriver PRIVATE ${LLVM_INCLUDE_DIRS})
target_compile_definitions(chaosDriver PRIVATE ${LLVM_DEFINITIONS})
# fixme llvm_libs are comint from chaosTransform
target_link_libraries(chaosDriver PRIVATE chaosTransform)
target_include_directories(chaosDriver PRIVATE ${CMAKE_SOURCE_DIR}/utils/chaos/include)