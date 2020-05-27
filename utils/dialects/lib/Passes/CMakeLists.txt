add_mlir_conversion_library(MLIRAthenaPasses
        RelationDestructorPass.cpp
        DeployDefaultFunctions.cpp
        BarrierLegalizerPass.cpp

        ADDITIONAL_HEADER_DIRS
        ${PROJECT_SOURCE_DIR}/include/

        DEPENDS
        MLIRAthenaGraph
        )

target_link_libraries(MLIRAthenaPasses
        PUBLIC
        MLIRIR
        MLIRPass
        MLIRAthenaGraph
        MLIRTransforms
        )
