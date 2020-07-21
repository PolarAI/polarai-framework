add_polar_library(polar_rt_support OBJECT 
        routines.cpp)
add_library(PolarAI::generic_backend_support ALIAS polar_rt_support)
target_include_directories(polar_rt_support PUBLIC 
                           $<BUILD_INTERFACE:${CMAKE_CURRENT_BINARY_DIR}>)
target_compile_definitions(polar_rt_support PRIVATE BUILD_DLL)

add_polar_library(polar_rt_support_static STATIC 
        $<TARGET_OBJECTS:polar_rt_support>)

install(TARGETS polar_rt_support_static EXPORT AthenaConfig
        RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR}
        COMPONENT Athena_Runtime
        LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
        COMPONENT Athena_Runtime
        NAMELINK_COMPONENT Athena_Development
        ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR}
        COMPONENT Athena_Development
        )
