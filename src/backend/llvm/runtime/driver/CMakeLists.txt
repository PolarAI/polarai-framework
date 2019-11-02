add_athena_library(runtime-driver STATIC runtime-driver.cpp ${GENERATOR_CPP})
target_link_libraries(runtime-driver PRIVATE athena-core)
add_dependencies(runtime-driver runtime-llvm-cpu generator_cpp)

target_include_directories(runtime-driver PRIVATE ${LLVM_INCLUDE_DIRS} ${CMAKE_CURRENT_SOURCE_DIR})
target_compile_definitions(runtime-driver PRIVATE ${LLVM_DEFINITIONS})

llvm_map_components_to_libnames(llvm_libs core irreader support)

target_link_libraries(runtime-driver PRIVATE ${llvm_libs})

if (UNIX AND NOT APPLE)
    target_link_libraries(runtime-driver PRIVATE dl.so)
elseif (WIN32)
    message(FATAL_ERROR "Runtime Driver is not supported by your OS")
endif ()
