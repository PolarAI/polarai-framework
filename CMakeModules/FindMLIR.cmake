set(MLIR_LIBS)

foreach (comp ${MLIR_FIND_COMPONENTS})
    find_library(__mlir_lib_${comp}
            NAMES MLIR${comp}
            PATHS ${LLVM_DIR} $ENV{LLVM_DIR}
            PATH_SUFFIXES lib)
    if (__mlir_lib_${comp})
        list(APPEND MLIR_LIBS ${__mlir_lib_${comp}})
    endif ()
endforeach ()

find_program(MLIR_TBLGEN
        mlir-tblgen
        PATHS ${LLVM_DIR} $ENV{LLVM_DIR}
        PATH_SUFFIXES bin)
find_program(MLIR_OPT
        mlir-opt
        PATHS ${LLVM_DIR} $ENV{LLVM_DIR}
        PATH_SUFFIXES bin)
find_path(MLIR_INCLUDE_DIRS mlir/Parser.h PATHS ${LLVM_DIR} $ENV{LLVM_DIR} PATH_SUFFIXES include)