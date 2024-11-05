find_package(GTest REQUIRED)

add_executable(math_tests   main.cpp math.cpp)
add_executable(matrix_tests main.cpp matrix.cpp)

target_link_libraries(math_tests   GTest::gtest GTest::gtest_main pthread math)
target_link_libraries(matrix_tests GTest::gtest GTest::gtest_main pthread math)

target_include_directories(math_tests   PRIVATE ${CMAKE_SOURCE_DIR}/math/include)
target_include_directories(matrix_tests PRIVATE ${CMAKE_SOURCE_DIR}/math/include)

gtest_discover_tests(math_tests   PROPERTIES ENVIRONMENT "GTEST_COLOR=1")
gtest_discover_tests(matrix_tests PROPERTIES ENVIRONMENT "GTEST_COLOR=1")
