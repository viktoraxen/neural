add_library(math STATIC
    src/math.cpp
    src/matrix.cpp
)

target_include_directories(math PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}/include
)

set_target_properties(math PROPERTIES
    ARCHIVE_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib
    LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib
    RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin
)

option(RUN_TESTS "Build the math library tests" OFF)

if(RUN_TESTS)
    add_subdirectory(test)
endif()
