configure_lit_site_cfg(
        ${CMAKE_CURRENT_SOURCE_DIR}/lit.site.cfg.py.in
        ${CMAKE_CURRENT_BINARY_DIR}/lit.site.cfg.py
        MAIN_CONFIG
        ${CMAKE_CURRENT_SOURCE_DIR}/lit.cfg.py
)

set(ATHENA_OPT_TEST_DEPENDS
        FileCheck count not
        athena-opt
        )

add_lit_testsuite(check-athena-opt "Running the athena-opt regression tests"
        ${CMAKE_CURRENT_BINARY_DIR}
        DEPENDS ${ATHENA_OPT_TEST_DEPENDS}
        )
set_target_properties(check-athena-opt PROPERTIES FOLDER "Tests")

add_lit_testsuites(ATHENA_OPT ${CMAKE_CURRENT_SOURCE_DIR} DEPENDS ${ATHENA_OPT_TEST_DEPENDS})
