find_package(MLIR COMPONENTS IR)

add_library(
        chaosDialects
        STATIC
        ClangDialect.cpp)

athena_disable_exceptions(chaosDialects)
athena_disable_rtti(chaosDialects)

if (UNIX)
    target_compile_options(chaosDialects PRIVATE -fPIC)
endif ()

generate_export_header(
        chaosDialects
        EXPORT_MACRO_NAME CHAOS_DIALECTS_EXPORT
        EXPORT_FILE_NAME ${CMAKE_BINARY_DIR}/include/chaos/Dialects/export.h
)

add_custom_command(OUTPUT ${CMAKE_BINARY_DIR}/include/chaos/Dialects/ClangDialect.h.inc
        ${CMAKE_CURRENT_BINARY_DIR}/ClangDialect.h.fake
        COMMAND ${MLIR_TBLGEN} --gen-op-decls -o=${CMAKE_BINARY_DIR}/include/chaos/Dialects/ClangDialect.h.inc
        -I=${MLIR_INCLUDE_DIRS} ${CMAKE_CURRENT_SOURCE_DIR}/ClangOps.td
        )

add_custom_command(OUTPUT ${CMAKE_BINARY_DIR}/include/chaos/Dialects/ClangDialect.cpp.inc
        ${CMAKE_CURRENT_BINARY_DIR}/ClangDialect.cpp.fake
        COMMAND ${MLIR_TBLGEN} --gen-op-defs -o=${CMAKE_BINARY_DIR}/include/chaos/Dialects/ClangDialect.cpp.inc
        -I=${MLIR_INCLUDE_DIRS} ${CMAKE_CURRENT_SOURCE_DIR}/ClangOps.td
        )

add_custom_target(chaos_dialects_tblgen
        DEPENDS ${CMAKE_BINARY_DIR}/include/chaos/Dialects/ClangDialect.h.inc
        ${CMAKE_CURRENT_BINARY_DIR}/ClangDialect.h.fake
        ${CMAKE_BINARY_DIR}/include/chaos/Dialects/ClangDialect.cpp.inc
        ${CMAKE_CURRENT_BINARY_DIR}/ClangDialect.cpp.fake
        SOURCES ${CMAKE_CURRENT_SOURCE_DIR}/ClangOps.td)

add_dependencies(chaosDialects chaos_dialects_tblgen)

# MLIR is full of weak symbols. Ubuntu is not doing well with them, so
# this is a hack to overcome the issue. Apparently, it produces lots of
# warning, but at least makes possible further development.
if (UNIX AND NOT APPLE)
    set(WHOLE_ARCHIVE -Wl,--exclude-libs,ALL -Wl,--whole-archive)
    set(NOWHOLE_ARCHIVE -Wl,--no-whole-archive)
endif ()

target_include_directories(chaosDialects PUBLIC ${LLVM_INCLUDE_DIRS})
target_link_libraries(chaosDialects PRIVATE ${WHOLE_ARCHIVE} ${MLIR_LIBS} ${NOWHOLE_ARCHIVE})
target_include_directories(chaosDialects PUBLIC ${CMAKE_SOURCE_DIR}/utils/chaos/include)
target_include_directories(chaosDialects PUBLIC ${CMAKE_BINARY_DIR}/include/chaos)