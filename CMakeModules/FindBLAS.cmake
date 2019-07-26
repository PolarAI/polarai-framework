function(__getboost)
    find_package(Boost)

    if (NOT ${BOOST_FOUND})
        include(${CMAKE_ROOT}/Modules/ExternalProject.cmake)
        externalproject_add(boost
                URL https://dl.bintray.com/boostorg/release/1.68.0/source/boost_1_68_0.tar.gz
                PREFIX ${PROJECT_SOURCE_DIR}/third_party
                CONFIGURE_COMMAND ""
                BUILD_COMMAND ""
                INSTALL_COMMAND ""
                )
        add_dependencies(runtime-llvm-cpu boost)
    endif ()

    target_include_directories(runtime-llvm-cpu PUBLIC ${BOOST_INCLUDE_DIRS})
endfunction(__getboost)

function(__getblis)
    include(${CMAKE_ROOT}/Modules/ExternalProject.cmake)
    find_program(MAKE_EXE NAMES gmake nmake make)
    externalproject_add(blis

            GIT_REPOSITORY "https://github.com/flame/blis.git"
            GIT_TAG "master"

            UPDATE_COMMAND ""
            PATCH_COMMAND ""

            PREFIX ${PROJECT_SOURCE_DIR}/third_party/
            INSTALL_DIR "blis/lib/"

            CONFIGURE_COMMAND ./configure --prefix=<INSTALL_DIR> --enable-cblas auto

            BUILD_IN_SOURCE TRUE
            BUILD_COMMAND ${MAKE_EXE}

            INSTALL_COMMAND ${MAKE_EXE} install
            )
    externalproject_get_property(blis INSTALL_DIR)
    add_dependencies(runtime-llvm-cpu blis)
    target_include_directories(runtime-llvm-cpu PUBLIC ${INSTALL_DIR}/include)
    target_compile_definitions(runtime-llvm-cpu PUBLIC BLIS)
    set(BLAS_LIBRARIES ${PROJECT_SOURCE_DIR}/libs/src/blis/include/blis/lib/libblis.a
            PARENT_SCOPE)
endfunction(__getblis)

option(FORCE_BLIS "Require library to use BLIS" OFF)
option(FORCE_ULAS "Require library to use uBLAS" OFF)

function(get_blas)
    if (FORCE_BLIS)
        message(STATUS "BLIS will be downloaded and compiled")
        CPMAddPackage(
                NAME BLIS
                GITHUB_REPOSITORY flame/blis
        )
    elseif (FORCE_UBLAS)
        message(STATUS "Boost will be downloaded and compiled")
        __getboost()
    else ()
        if (APPLE)
            set(BLA_VENDOR Apple)
            find_package(BLAS CONFIG)
            #        target_compile_definitions(runtime-llvm-cpu PUBLIC -DAPPLE_ACCELERATE)
            message(STATUS "Apple Accelerate Framework will be used")
        else ()
            message(STATUS "No BLAS installation found. Boost will be downloaded and compiled")
            getboost()
        endif ()
    endif ()
endfunction(get_blas)