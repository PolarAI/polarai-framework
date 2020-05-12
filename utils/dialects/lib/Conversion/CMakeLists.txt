add_mlir_conversion_library(MLIRAthenaConversion
        GraphToRuntimePass.cpp
        ArgInfo.cpp

        ADDITIONAL_HEADER_DIRS
        ${PROJECT_SOURCE_DIR}/include/

        DEPENDS
        MLIRAthenaGraph
)

target_link_libraries(MLIRAthenaConversion
        PUBLIC
        MLIRIR
        MLIRPass
        MLIRAthenaGraph
        MLIRTransforms
)