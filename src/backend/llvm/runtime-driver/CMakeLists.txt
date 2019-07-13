add_athena_library(runtime-driver STATIC runtime-driver.cpp ${GENERATOR_CPP})
target_link_libraries(runtime-driver PRIVATE athena-core)
add_dependencies(runtime-driver runtime-llvm-cpu generator_cpp)

target_include_directories(runtime-driver PRIVATE ${LLVM_INCLUDE_DIRS})
target_compile_definitions(runtime-driver PRIVATE ${LLVM_DEFINITIONS})

llvm_map_components_to_libnames(llvm_libs all)

target_link_libraries(runtime-driver PRIVATE ${llvm_libs})

if (UNIX AND NOT APPLE)
    target_link_libraries(runtime-driver dl.so)
#else ()
#    message(FATAL_ERROR "Dynamic link library does not defined for your OS")
endif ()
