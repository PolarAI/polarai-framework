if(POLICY CMP0068)
    cmake_policy(SET CMP0068 NEW)
    set(CMAKE_BUILD_WITH_INSTALL_NAME_DIR ON)
endif()

if(POLICY CMP0075)
    cmake_policy(SET CMP0075 NEW)
endif()

if(POLICY CMP0077)
    cmake_policy(SET CMP0077 NEW)
endif()

find_package(LLVM REQUIRED CONFIG)

list(APPEND CMAKE_MODULE_PATH "${LLVM_CMAKE_DIR}")
include(TableGen)
include(AddLLVM)

set(LLVM_BUILD_BINARY_DIR ${LLVM_BINARY_DIR}/bin)

function(add_device_target target_name triple)

  file(MAKE_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}/tmp)

  if (triple MATCHES "spirv")
    set(target_triple "spir")
  else()
    set(target_triple ${triple})
  endif()

  foreach(src IN LISTS ARGN)
    get_filename_component(filename ${src} NAME)
    add_custom_command(OUTPUT "${CMAKE_CURRENT_BINARY_DIR}/tmp/${filename}.ll" 
      COMMAND ${LLVM_BUILD_BINARY_DIR}/clang -cc1 -finclude-default-header 
      -x cl -cl-std=CL2.0 -triple ${target_triple}
      -I ${PROJECT_SOURCE_DIR}/utils/kernels/include
      -emit-llvm -o "${CMAKE_CURRENT_BINARY_DIR}/tmp/${filename}.ll" "${src}"
      DEPENDS ${src})
    add_custom_command(OUTPUT "${CMAKE_CURRENT_BINARY_DIR}/tmp/${filename}.bc"
      COMMAND ${LLVM_BUILD_BINARY_DIR}/llvm-as 
      "${CMAKE_CURRENT_BINARY_DIR}/tmp/${filename}.ll"
      -o "${CMAKE_CURRENT_BINARY_DIR}/tmp/${filename}.bc"
      DEPENDS "${CMAKE_CURRENT_BINARY_DIR}/tmp/${filename}.ll")
    list(APPEND COMPILED_BC "${CMAKE_CURRENT_BINARY_DIR}/tmp/${filename}.bc")
  endforeach()

  string(REPLACE ";" " " STRBC ${COMPILED_BC})
  add_custom_command(OUTPUT "${CMAKE_CURRENT_BINARY_DIR}/tmp/${target_name}.bc"
    COMMAND ${LLVM_BUILD_BINARY_DIR}/llvm-link ${STRBC}
    -o ${CMAKE_CURRENT_BINARY_DIR}/tmp/${target_name}.bc
    DEPENDS ${COMPILED_BC})

  if (triple MATCHES "spirv")
    set(outfile ${CMAKE_CURRENT_BINARY_DIR}/${target_name}.spv)
    add_custom_command(OUTPUT ${outfile}
      COMMAND ${LLVM_BUILD_BINARY_DIR}/llvm-spirv 
      "${CMAKE_CURRENT_BINARY_DIR}/tmp/${target_name}.bc"
      -s -o ${outfile}
      DEPENDS "${CMAKE_CURRENT_BINARY_DIR}/tmp/${target_name}.bc")
  else()
    set(outfile ${CMAKE_CURRENT_BINARY_DIR}/${target_name})
    add_custom_command(OUTPUT ${outfile}
      COMMAND ${LLVM_BUILD_BINARY_DIR}/llc
      "${CMAKE_CURRENT_BINARY_DIR}/tmp/${target_name}.bc"
      -o ${outfile}
      DEPENDS "${CMAKE_CURRENT_BINARY_DIR}/tmp/${target_name}.bc")
  endif()
    
  add_custom_target(${target_name} DEPENDS ${outfile})
  set_target_properties(${target_name} PROPERTIES BINARY_FILE ${outfile})
endfunction()

add_subdirectory(src)
