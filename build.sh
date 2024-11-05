#!/bin/bash

RED='\033[0;31m'
GREEN='\033[0;32m'
NC='\033[0m' # No Color

# Set the build directory
BUILD_DIR="build"

# Function to exit the script if a command fails
function exit_on_failure() {
    if [ $? -ne 0 ]; then
        printf "${RED}X\n"
        echo "Error: $1 failed. Exiting."
        exit 1
    fi
}

MATH_LIB_TESTS=OFF
USE_VALGRIND=OFF
RUN_NEURAL=OFF
VERBOSE=OFF

# Parse command-line arguments
while [[ "$#" -gt 0 ]]; do
    case $1 in
        -t|--testing)      RUN_TESTS=ON ;; # Enable testing if the flag is passed
        -m|--memory-check) USE_VALGRIND=ON ;;
        -r|--run)          RUN_NEURAL=ON ;;     # Enable running the main program if the flag is passed
        -v|--verbose)      VERBOSE=ON ;;        # Enable verbose output if the flag is passed
        *) echo "Unknown parameter passed: $1"; exit 1 ;;
    esac
    shift
done

rm -rf $BUILD_DIR

if [ ! -d "$BUILD_DIR" ]; then
    mkdir $BUILD_DIR
fi

cd $BUILD_DIR

printf "Running CMake...    "
if [ $VERBOSE == "ON" ]; then
    cmake -S .. -DRUN_TESTS=$RUN_TESTS
else
    cmake -S .. -DRUN_TESTS=$RUN_TESTS > /dev/null
fi
exit_on_failure "CMake"
printf "${GREEN}O\n${NC}"

printf "Building Neural...  "
if [ $VERBOSE == "ON" ]; then
    make --silent -j$(nproc)   # or just use `make` if you don't want parallel builds
else
    make --silent -j$(nproc) > /dev/null
fi
exit_on_failure "Make"
printf "${GREEN}O\n${NC}"

if [ $RUN_TESTS == "ON" ]; then
    echo "Running tests...   "
    ctest --color --output-on-failure --progress
    exit_on_failure "Tests"
fi

if [ $USE_VALGRIND == "ON" ]; then
    echo "Running Valgrind..."
    valgrind --leak-check=full --show-leak-kinds=all --track-origins=yes ./neural
    exit_on_failure "Valgrind"
fi

if [ $RUN_NEURAL == "ON" ]; then
    echo "Running Neural..."
    ./neural
    exit_on_failure "Neural"
fi
