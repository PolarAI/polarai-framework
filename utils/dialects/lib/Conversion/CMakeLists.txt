add_mlir_conversion_library(MLIRAthenaConversion
        GraphToRuntimePass.cpp

        ADDITIONAL_HEADER_DIRS
        ${PROJECT_SOURCE_DIR}/include/

        DEPENDS
        MLIRAthenaRuntime
        MLIRAthenaGraph
)

target_link_libraries(MLIRAthenaConversion
        PUBLIC
        MLIRIR
        MLIRPass
        MLIRAthenaRuntime
        MLIRAthenaGraph
        MLIRTransforms
)