set(ATH_RT_LLVM_DRIVER "AthenaLLVMRuntimeDriver" CACHE STRING "" FORCE)

add_athena_library(${ATH_RT_LLVM_DRIVER} STATIC runtime-driver.cpp ${GENERATOR_CPP})
target_link_libraries(${ATH_RT_LLVM_DRIVER} PRIVATE ${ATH_CORE})
add_dependencies(${ATH_RT_LLVM_DRIVER} generator_cpp)

target_include_directories(${ATH_RT_LLVM_DRIVER} PRIVATE ${LLVM_INCLUDE_DIRS} ${CMAKE_CURRENT_SOURCE_DIR})
target_compile_definitions(${ATH_RT_LLVM_DRIVER} PRIVATE ${LLVM_DEFINITIONS})

llvm_map_components_to_libnames(llvm_libs core irreader support)

target_link_libraries(${ATH_RT_LLVM_DRIVER} PRIVATE ${llvm_libs})

if (UNIX AND NOT APPLE)
    target_link_libraries(${ATH_RT_LLVM_DRIVER} PRIVATE dl.so)
elseif (WIN32)
    message(FATAL_ERROR "Runtime Driver is not supported by your OS")
endif ()
