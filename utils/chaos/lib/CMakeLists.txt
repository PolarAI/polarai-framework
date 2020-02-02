include(GenerateExportHeader)
find_package(LLVM CONFIG)

set(CMAKE_CXX_VISIBILITY_PRESET hidden)
set(CMAKE_VISIBILITY_INLINES_HIDDEN 1)

add_subdirectory(Target)
add_subdirectory(Transform)
add_subdirectory(Driver)
add_subdirectory(Frontend)
add_subdirectory(Dialects)