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

# add_test(NAME Tests COMMAND tests)
gtest_discover_tests(tests)
