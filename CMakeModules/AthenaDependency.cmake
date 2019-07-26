if (EXISTS ${CMAKE_BINARY_DIR}/conanbuildinfo.cmake)
    include(${CMAKE_BINARY_DIR}/conanbuildinfo.cmake)
    conan_basic_setup(TARGETS)
endif ()

include(FetchContent)
include(CMakeParseArguments)
set(FETCHCONTENT_QUIET off)

set(oneValueArgs
        TARGET_NAME
        GIT_URL
        GIT_TAG
        PACKAGE
        INCLUDE_DIRS
        INCLUDE_PATH
        )

set(multiValueArgs
        LIBRARIES
        COMPONENTS
        )

function(AthenaAddDependency)
    cmake_parse_arguments(PARSED_ARGS "" "${oneValueArgs}" "${multiValueArgs}" ${ARGN})

    set(NEEDS_DOWNLOAD TRUE)

    if (TARGET CONAN::${PARSED_ARGS_TARGET_NAME})
        add_library(AthenaDep::${PARSED_ARGS_TARGET_NAME} ALIAS CONAN::${PARSED_ARGS_TARGET_NAME})
        set(NEEDS_DOWNLOAD FALSE)
    elseif (PARSED_ARGS_PACKAGE)
        if (PARSED_ARGS_COMPONENTS)
            find_package(${PARSED_ARGS_PACKAGE}
                    COMPONENTS ${PARSED_ARGS_COMPONENTS})
        else ()
            find_package(${PARSED_ARGS_PACKAGE})
        endif ()
        if (${PARSED_ARGS_PACKAGE}_FOUND)
            set(NEEDS_DOWNLOAD FALSE)
            add_library(AthenaDep::${PARSED_ARGS_TARGET_NAME} INTERFACE IMPORTED)

            set(INCLUDE_DIRS)

            if (${PARSED_ARGS_PACKAGE}_INCLUDE_DIRS)
                set(INCLUDE_DIRS ${${PARSED_ARGS_PACKAGE}_INCLUDE_DIRS})
                target_include_directories(AthenaDep::${PARSED_ARGS_TARGET_NAME}
                        PUBLIC INTERFACE ${${PARSED_ARGS_PACKAGE}_INCLUDE_DIRS})
            endif ()

            if (PARSED_ARGS_LIBRARIES)
                foreach (LIB ${PARSED_ARGS_LIBRARIES})
                    add_library(AthenaDep::${PARSED_ARGS_TARGET_NAME}-${LIB} INTERFACE IMPORTED)
                    target_link_libraries(AthenaDep::${PARSED_ARGS_TARGET_NAME}-${LIB} PUBLIC INTERFACE ${LIB})
                    target_link_libraries(AthenaDep::${PARSED_ARGS_TARGET_NAME} PUBLIC INTERFACE AthenaDep::${PARSED_ARGS_TARGET_NAME}-${LIB})
                    if (INCLUDE_DIRS)
                        target_include_directories(AthenaDep::${PARSED_ARGS_TARGET_NAME}-${LIB} PUBLIC INTERFACE ${INCLUDE_DIRS})
                    endif ()
                endforeach ()
            else ()
                target_link_libraries(AthenaDep::${PARSED_ARGS_TARGET_NAME}
                        PUBLIC INTERFACE ${${PARSED_ARGS_PACKAGE}_LIBRARIES})
            endif ()
            if (INCLUDE_DIRS)
                target_include_directories(AthenaDep::${PARSED_ARGS_TARGET_NAME} PUBLIC INTERFACE ${INCLUDE_DIRS})
            endif ()
        endif ()
    endif ()

    if (NEEDS_DOWNLOAD)
        if (PARSED_ARGS_GIT_URL)
            set(GIT_TAG master)
            if (PARSED_ARGS_GIT_TAG)
                set(GIT_TAG ${PARSED_ARGS_GIT_TAG})
            endif ()
            message(${PARSED_ARGS_GIT_URL})
            FetchContent_Declare(${PARSED_ARGS_TARGET_NAME}
                    GIT_REPOSITORY ${PARSED_ARGS_GIT_URL}
                    GIT_TAG ${PARSED_ARGS_GIT_TAG})
        endif ()

        FetchContent_GetProperties("${PARSED_ARGS_TARGET_NAME}")
        if (NOT ${PARSED_ARGS_TARGET_NAME}_POPULATED)
            FetchContent_Populate("${PARSED_ARGS_TARGET_NAME}")
            add_subdirectory(${${PARSED_ARGS_TARGET_NAME}_SOURCE_DIR} ${${PARSED_ARGS_TARGET_NAME}_BINARY_DIR})
            add_library(AthenaDep::${PARSED_ARGS_TARGET_NAME} INTERFACE IMPORTED)

            set(INCLUDE_DIRS)
            if (PARSED_ARGS_INCLUDE_DIRS)
                set(INCLUDE_DIRS ${${PARSED_ARGS_INCLUDE_DIRS}})
            elseif (PARSED_ARGS_INCLUDE_PATH)
                set(INCLUDE_DIRS ${${PARSED_ARGS_TARGET_NAME}_SOURCE_DIR}/${PARSED_ARGS_INCLUDE_PATH})
            else ()
                set(INCLUDE_DIRS ${${PARSED_ARGS_TARGET_NAME}_SOURCE_DIR}/include)
            endif ()
            if (PARSED_ARGS_LIBRARIES)
                foreach (LIB ${PARSED_ARGS_LIBRARIES})
                    add_library(AthenaDep::${PARSED_ARGS_TARGET_NAME}-${LIB} INTERFACE IMPORTED)
                    target_include_directories(AthenaDep::${PARSED_ARGS_TARGET_NAME}-${LIB} PUBLIC INTERFACE ${INCLUDE_DIRS})
                    target_link_libraries(AthenaDep::${PARSED_ARGS_TARGET_NAME}-${LIB} PUBLIC INTERFACE ${LIB})
                    target_link_libraries(AthenaDep::${PARSED_ARGS_TARGET_NAME} PUBLIC INTERFACE AthenaDep::${PARSED_ARGS_TARGET_NAME}-${LIB})
                endforeach ()
            else ()
                target_link_libraries(AthenaDep::${PARSED_ARGS_TARGET_NAME} INTERFACE ${PARSED_ARGS_TARGET_NAME})
                target_include_directories(AthenaDep::${PARSED_ARGS_TARGET_NAME} INTERFACE ${INCLUDE_DIRS})
            endif ()
        endif ()
    endif ()
endfunction()