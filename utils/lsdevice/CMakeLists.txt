add_athena_executable(lsdevice main.cpp)

target_include_directories(lsdevice PRIVATE 
                          ${PROJECT_SOURCE_DIR}/src/backend/llvm/runtime/driver)
target_link_libraries(lsdevice PRIVATE ${ATH_RT_LLVM_DRIVER})
