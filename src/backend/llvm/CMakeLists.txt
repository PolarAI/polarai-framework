find_package(LLVM CONFIG)
find_package(PythonInterp REQUIRED)

message(STATUS "Found LLVM ${LLVM_PACKAGE_VERSION}")
message(STATUS "Using LLVMConfig.cmake in: ${LLVM_DIR}")

set(WRAPPER_CPP ${CMAKE_CURRENT_BINARY_DIR}/wrapper.cpp)
set(GENERATOR_CPP ${CMAKE_CURRENT_BINARY_DIR}/generator.cpp)

file(WRITE ${WRAPPER_CPP} "")
file(WRITE ${GENERATOR_CPP} "")

add_custom_target(wrapper_cpp
        COMMAND ${PYTHON_EXECUTABLE} ${CMAKE_CURRENT_SOURCE_DIR}/tablegen.py
        ${CMAKE_CURRENT_SOURCE_DIR}/builtins.td ${WRAPPER_CPP} wrapper
        )

add_custom_target(generator_cpp
        COMMAND ${PYTHON_EXECUTABLE}
        ${CMAKE_CURRENT_SOURCE_DIR}/tablegen.py
        ${CMAKE_CURRENT_SOURCE_DIR}/builtins.td ${GENERATOR_CPP} driver
        )

add_subdirectory(runtime-cpu)
add_subdirectory(runtime-driver)

file(GLOB codegen_src "codegen/*.cpp")

add_athena_library(backend-llvm STATIC
        LLVMExecutor.cpp
        LLVMGenerator.cpp
        AthenaJIT.cpp
        LLVMTrivialAllocator.cpp
        utils.cpp
        ${codegen_src})

target_include_directories(backend-llvm PUBLIC ${LLVM_INCLUDE_DIRS})
target_compile_definitions(backend-llvm PUBLIC ${LLVM_DEFINITIONS})

llvm_map_components_to_libnames(llvm_libs core irreader support transformutils passes codegen orcjit x86asmparser x86codegen)

target_link_libraries(backend-llvm PUBLIC runtime-driver ${llvm_libs})
add_dependencies(backend-llvm runtime-llvm-cpu)
# TODO make JIT link against executable
target_link_libraries(backend-llvm PUBLIC MemoryLoader)
