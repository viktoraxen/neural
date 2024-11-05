cmake_minimum_required(VERSION 3.10)

project(Neural VERSION 1.0)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED True)

set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

enable_testing()

add_subdirectory(math)

add_executable(neural 
    src/main.cpp
    src/network.cpp
    src/layer.cpp
)

target_include_directories(neural PRIVATE
    ${PROJECT_SOURCE_DIR}/include
)

target_link_libraries(neural PRIVATE 
    math
)

option(RUN_TESTS "Build the math library tests" OFF)

if (RUN_TESTS)
    add_subdirectory(test)
endif()
