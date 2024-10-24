#!/bin/bash

# Set the build directory
BUILD_DIR="build"

# Function to exit the script if a command fails
function exit_on_failure() {
    if [ $? -ne 0 ]; then
        echo "Error: $1 failed. Exiting."
        exit 1
    fi
}

MATH_LIB_TESTS=OFF
RUN_NEURAL=OFF
VERBOSE=OFF

# Parse command-line arguments
while [[ "$#" -gt 0 ]]; do
    case $1 in
        -t|--testing) MATH_LIB_TESTS=ON ;; # Enable testing if the flag is passed
        -r|--run)     RUN_NEURAL=ON ;;     # Enable running the main program if the flag is passed
        -v|--verbose) VERBOSE=ON ;;        # Enable verbose output if the flag is passed
        *) echo "Unknown parameter passed: $1"; exit 1 ;;
    esac
    shift
done

rm -rf $BUILD_DIR

if [ ! -d "$BUILD_DIR" ]; then
    mkdir $BUILD_DIR
fi

cd $BUILD_DIR

echo "Running CMake..."
if [ $VERBOSE == "ON" ]; then
    cmake -S .. -DMATH_LIB_TESTS=$MATH_LIB_TESTS
else
    cmake -S .. -DMATH_LIB_TESTS=$MATH_LIB_TESTS > /dev/null
fi
exit_on_failure "CMake"

echo "Building Neural..."
if [ $VERBOSE == "ON" ]; then
    make --silent -j$(nproc)   # or just use `make` if you don't want parallel builds
else
    make --silent -j$(nproc) > /dev/null
fi
exit_on_failure "Make"

if [ $MATH_LIB_TESTS == "ON" ]; then
    echo "Running tests..."
    ctest --color --output-on-failure
    exit_on_failure "Tests"
fi

if [ $RUN_NEURAL == "ON" ]; then
    echo "Running Neural..."
    ./neural
    exit_on_failure "Neural"
fi
