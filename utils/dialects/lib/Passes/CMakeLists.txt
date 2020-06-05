add_mlir_conversion_library(MLIRAthenaPasses
        RelationDestructorPass.cpp
        DeployDefaultFunctions.cpp
        BarrierLegalizerPass.cpp
        LegalizeRTForLoweringPass.cpp
        ReleaseDependencyPass.cpp

        ADDITIONAL_HEADER_DIRS
        ${PROJECT_SOURCE_DIR}/include/
        ${PROJECT_BINARY_DIR}/include

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
