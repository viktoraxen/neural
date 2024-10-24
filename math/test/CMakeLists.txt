find_package(GTest REQUIRED)

add_executable(tests
    main.cpp
    math.cpp
    matrix.cpp
)

target_link_libraries(tests 
    GTest::gtest 
    GTest::gtest_main 
    pthread 
    math
)

set(VALGRIND_COMMAND 
    valgrind 
    --leak-check=full 
    --track-origins=yes 
    --error-exitcode=1
)

# add_test(NAME Tests COMMAND tests)
gtest_discover_tests(tests
    PROPERTIES
        ENVIRONMENT "GTEST_COLOR=1"
        # TEST_PREFIX "Valgrind:"
        # LABELS "Valgrind"
)

# add_custom_target(run_tests_with_valgrind
#     COMMAND ${VALGRIND_COMMAND} ./tests
#     DEPENDS tests
# )
