add_mlir_conversion_library(MLIRAthenaPasses
        LoaderFunctionAnalysis.cpp
        DeployDefaultFunctions.cpp

        ADDITIONAL_HEADER_DIRS
        ${PROJECT_SOURCE_DIR}/include/

        DEPENDS
        MLIRAthenaRuntime
        MLIRAthenaGraph
        )

target_link_libraries(MLIRAthenaPasses
        PUBLIC
        MLIRIR
        MLIRPass
        MLIRAthenaRuntime
        MLIRAthenaGraph
        MLIRTransforms
        )
