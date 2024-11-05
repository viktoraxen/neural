find_package(GTest REQUIRED)

add_executable(network_tests 
    main.cpp 
    network.cpp
    ${CMAKE_SOURCE_DIR}/src/network.cpp
    ${CMAKE_SOURCE_DIR}/src/layer.cpp
)

target_link_libraries(network_tests GTest::gtest GTest::gtest_main pthread math)

target_include_directories(network_tests PRIVATE 
    ${CMAKE_SOURCE_DIR}/include
    ${CMAKE_SOURCE_DIR}/math/include
)

gtest_discover_tests(network_tests PROPERTIES ENVIRONMENT "GTEST_COLOR=1")
