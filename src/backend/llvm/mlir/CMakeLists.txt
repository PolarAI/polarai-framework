set(ATH_BACKEND_LLVM_MLIR "AthenaMlir" CACHE STRING "" FORCE)
add_athena_library(
        ${ATH_BACKEND_LLVM_MLIR} STATIC
        ATH_MLIR
        backend/llvm/mlir/mlir_export.h
        GraphDialect.cpp
        MLIRGenerator.cpp
)

file(MAKE_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}/tablegen)

add_custom_command(OUTPUT ${CMAKE_CURRENT_BINARY_DIR}/tablegen/GraphDialect.h.inc
        ${CMAKE_CURRENT_BINARY_DIR}/tablegen/GraphDialect.h.fake
        COMMAND ${MLIR_TBLGEN} --gen-op-decls -o=${CMAKE_CURRENT_BINARY_DIR}/tablegen/GraphDialect.h.inc
        -I=${MLIR_INCLUDE_DIRS} ${CMAKE_CURRENT_SOURCE_DIR}/GraphDialect.td
        )
add_custom_command(OUTPUT ${CMAKE_SOURCE_DIR}/docs/design/llvm/02_mlir_graph_reference.md
        ${CMAKE_SOURCE_DIR}/docs/design/llvm/02_mlir_graph_reference.md.fake
        COMMAND ${MLIR_TBLGEN} --gen-op-doc -o=${CMAKE_SOURCE_DIR}/docs/design/llvm/02_mlir_graph_reference.md
        -I=${MLIR_INCLUDE_DIRS} ${CMAKE_CURRENT_SOURCE_DIR}/GraphDialect.td
        )
add_custom_command(OUTPUT ${CMAKE_CURRENT_BINARY_DIR}/tablegen/GraphDialect.cpp.inc
        ${CMAKE_CURRENT_BINARY_DIR}/tablegen/GraphDialect.cpp.fake
        COMMAND ${MLIR_TBLGEN} --gen-op-defs -o=${CMAKE_CURRENT_BINARY_DIR}/tablegen/GraphDialect.cpp.inc
        -I=${MLIR_INCLUDE_DIRS} ${CMAKE_CURRENT_SOURCE_DIR}/GraphDialect.td
        )
add_custom_target(mlir_dialect_tblgen
        DEPENDS ${CMAKE_CURRENT_BINARY_DIR}/tablegen/GraphDialect.h.inc
        ${CMAKE_CURRENT_BINARY_DIR}/tablegen/GraphDialect.h.fake
        ${CMAKE_SOURCE_DIR}/docs/design/llvm/02_mlir_graph_reference.md
        ${CMAKE_SOURCE_DIR}/docs/design/llvm/02_mlir_graph_reference.md.fake
        ${CMAKE_CURRENT_BINARY_DIR}/tablegen/GraphDialect.cpp.inc
        ${CMAKE_CURRENT_BINARY_DIR}/tablegen/GraphDialect.cpp.fake
        SOURCES ${CMAKE_CURRENT_SOURCE_DIR}/GraphDialect.td)

add_dependencies(${ATH_BACKEND_LLVM_MLIR} mlir_dialect_tblgen)
target_link_libraries(${ATH_BACKEND_LLVM_MLIR} PRIVATE ${MLIR_LIBS} ${AthenaCore})
target_include_directories(${ATH_BACKEND_LLVM_MLIR} PUBLIC ${MLIR_INCLUDE_DIRS}
        PRIVATE ${CMAKE_CURRENT_BINARY_DIR}/tablegen)