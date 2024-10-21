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

# Step 1: Clean previous build (optional, uncomment if you want a fresh build every time)
# echo "Cleaning previous build..."
rm -rf $BUILD_DIR

# Step 2: Create the build directory if it doesn't exist
if [ ! -d "$BUILD_DIR" ]; then
    echo "Creating build directory..."
    mkdir $BUILD_DIR
fi

# Step 3: Navigate to the build directory
cd $BUILD_DIR

# Step 4: Run CMake
echo "Running CMake..."
cmake --log-level=WARNING ..
exit_on_failure "CMake"

# Step 5: Run Make (or use the number of cores for faster builds)
echo "Building the project..."
make --silent -j$(nproc)   # or just use `make` if you don't want parallel builds
exit_on_failure "Make"

# Step 6: Run tests with CTest
echo "Running tests..."
ctest --color --output-on-failure
exit_on_failure "Tests"

# Step 7: Run Valgrind tests
# echo "Running Valgrind tests..."
# valgrind --leak-check=full --quiet ./bin/*
# exit_on_failure "Valgrind tests"

# Optional: You can add a message for successful completion
echo "Build and tests completed successfully!"

