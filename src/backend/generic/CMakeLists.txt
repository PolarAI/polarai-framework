add_subdirectory(runtime)

if (NOT POLAR_NO_LLVM)
  add_subdirectory(lib)
endif()