cmake_minimum_required(VERSION 3.10)

project(Neural VERSION 1.0)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED True)

include_directories(include)

add_library(math STATIC src/math.cpp)

set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

enable_testing()

# Download and configure GoogleTest
# include(FetchContent)
#
# FetchContent_Declare(
#   googletest
#   URL https://github.com/google/googletest/archive/refs/tags/release-1.12.1.zip  # Latest stable version
# )
#
# # Download the content
# FetchContent_MakeAvailable(googletest)

find_package(GTest REQUIRED)

add_executable(tests test/main.cpp)
target_link_libraries(tests GTest::gtest GTest::gtest_main pthread math)

add_test(NAME Tests COMMAND tests)

# find_program(valgrind "valgrind")
#
# if(valgrind)
#     add_test(name matrixvalgrindtest command ${valgrind} --leak-check=full ./matrix_test)
# endif()
