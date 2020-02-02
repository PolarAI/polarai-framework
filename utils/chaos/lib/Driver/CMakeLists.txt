add_library(
        chaosDriver
        SHARED
        Driver.cpp
)
athena_disable_exceptions(chaosDriver)
athena_disable_rtti(chaosDriver)
if (UNIX)
    target_compile_options(chaosDriver PRIVATE -fPIC)
endif ()

generate_export_header(
        chaosDriver
        EXPORT_MACRO_NAME CHAOS_DRIVER_EXPORT
        EXPORT_FILE_NAME ${CMAKE_BINARY_DIR}/include/chaos/Driver/export.h
)

llvm_map_components_to_libnames(llvm_libs
        support)

target_include_directories(chaosDriver PUBLIC ${LLVM_INCLUDE_DIRS})
target_compile_definitions(chaosDriver PUBLIC ${LLVM_DEFINITIONS})
target_link_libraries(chaosDriver PRIVATE ${llvm_libs} chaosTransform chaosTarget chaosFrontend)
target_include_directories(chaosDriver PUBLIC ${CMAKE_SOURCE_DIR}/utils/chaos/include)
target_include_directories(chaosDriver PUBLIC ${CMAKE_BINARY_DIR}/include/chaos)