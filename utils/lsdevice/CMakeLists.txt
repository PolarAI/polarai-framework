add_athena_executable(lsdevice main.cpp)

target_include_directories(lsdevice PRIVATE 
                          ${PROJECT_SOURCE_DIR}/src/backend/llvm/runtime/driver)
if (APPLE)
    set(WHOLE_ARCHIVE -Wl,-all_load)
    set(NOWHOLE_ARCHIVE)
elseif (UNIX)
    set(WHOLE_ARCHIVE -Wl,--whole-archive)
    set(NOWHOLE_ARCHIVE -Wl,--no-whole-archive)
endif ()
target_link_libraries(lsdevice PRIVATE ${ATH_RT_LLVM_DRIVER} 
                      ${WHOLE_ARCHIVE} ${ATH_UTILS} ${NOWHOLE_ARCHIVE})
