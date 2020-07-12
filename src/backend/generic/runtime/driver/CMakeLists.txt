set(ATH_RT_LLVM_DRIVER "AthenaLLVMRTDriver" CACHE STRING "" FORCE)

configure_file(config.h.in ${CMAKE_CURRENT_BINARY_DIR}/config.h @ONLY)

add_athena_library(${ATH_RT_LLVM_DRIVER} STATIC
  DONTCARE
  dontcare.h
  DynamicLibrary_posix.cpp
  RuntimeDriver.cpp)

target_include_directories(${ATH_RT_LLVM_DRIVER} PRIVATE 
                           ${CMAKE_CURRENT_BINARY_DIR})

target_link_libraries(${ATH_RT_LLVM_DRIVER} PRIVATE dl)

install(TARGETS ${ATH_RT_LLVM_DRIVER} EXPORT AthenaConfig
        RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR}
        COMPONENT Athena_Runtime
        LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
        COMPONENT Athena_Runtime
        NAMELINK_COMPONENT Athena_Development
        ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR}
        COMPONENT Athena_Development
        )
